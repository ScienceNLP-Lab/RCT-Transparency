import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from shared.utils import batched_index_select

from transformers import BertPreTrainedModel, BertModel

import logging

logger = logging.getLogger('root')

class BertForEntity(BertPreTrainedModel):
    def __init__(
            self, 
            config, 
            num_entity_labels,
            head_hidden_dim=150, 
            width_embedding_dim=150, 
            max_span_length=8, 
            sent_pos_embedding_dim=150,
            add_relative_position=False,
            add_special_tokens=False,
    ):
        super().__init__(config)

        self.bert = BertModel(config)
        self.hidden_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.width_embedding = nn.Embedding(max_span_length+1, width_embedding_dim)
        self.num_ner_labels = num_entity_labels

        self.add_relative_position = add_relative_position
        self.add_special_tokens = add_special_tokens

        if self.add_relative_position:
            self.sent_pos_embedding = nn.Embedding(sent_pos_embedding_dim+1, width_embedding_dim)

        if self.add_special_tokens:
            input_dim = config.hidden_size*3+width_embedding_dim
        elif self.add_relative_position:
            input_dim = config.hidden_size*2+width_embedding_dim*2
        else:
            input_dim = config.hidden_size*2+width_embedding_dim

        self.entity_classifier = nn.Sequential(
            nn.Linear(input_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(head_hidden_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(head_hidden_dim, self.num_ner_labels)
        )
        self.init_weights()

    def _get_span_embeddings(
            self, 
            input_ids, 
            spans, 
            token_type_ids=None, 
            attention_mask=None, 
            sent_header_indices=None, 
            sent_pos_in_doc=None
    ):
        # sequence_output, pooled_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.hidden_dropout(sequence_output)

        """
        spans: [batch_size, num_spans, 3]; 0: left_end, 1: right_end, 2: width
        spans_mask: (batch_size, num_spans, )
        """
        spans_start = spans[:, :, 0].view(spans.size(0), -1)
        spans_start_embedding = batched_index_select(sequence_output, spans_start)
        spans_end = spans[:, :, 1].view(spans.size(0), -1)
        spans_end_embedding = batched_index_select(sequence_output, spans_end)

        spans_width = spans[:, :, 2].view(spans.size(0), -1)
        spans_width_embedding = self.width_embedding(spans_width)

        if self.add_special_tokens:
            sent_header_indices_ = sent_header_indices.view(sent_header_indices.size(0), -1)
            sent_header_embedding = batched_index_select(sequence_output, sent_header_indices_)

        if self.add_relative_position:
            sent_pos_in_doc_ = sent_pos_in_doc.view(sent_pos_in_doc.size(0), -1)
            sent_pos_embedding = self.sent_pos_embedding(sent_pos_in_doc_)

        if self.add_special_tokens:
            # Concatenate embeddings of left/right points and the width embedding
            spans_embedding = torch.cat((
                spans_start_embedding, 
                spans_end_embedding, 
                spans_width_embedding,
                sent_header_embedding
                # sent_pos_embedding
            ), dim=-1)

        elif self.add_relative_position:
            spans_embedding = torch.cat((
                spans_start_embedding, 
                spans_end_embedding, 
                spans_width_embedding,
                sent_pos_embedding
            ), dim=-1)            

        else:
            spans_embedding = torch.cat((
                spans_start_embedding, 
                spans_end_embedding, 
                spans_width_embedding
            ), dim=-1)

        """
        spans_embedding: (batch_size, num_spans, hidden_size*2+embedding_dim)
        """
        return spans_embedding


    def forward(self, input_ids, spans, spans_mask, spans_ner_label=None, token_type_ids=None, attention_mask=None, sent_header_indices=None, sent_pos_in_doc=None):
        spans_embedding = self._get_span_embeddings(input_ids, spans, token_type_ids=token_type_ids, attention_mask=attention_mask, sent_header_indices=sent_header_indices, sent_pos_in_doc=sent_pos_in_doc)

        ffnn_hidden_entity = []
        hidden_entity = spans_embedding
        for layer in self.entity_classifier:
            hidden_entity = layer(hidden_entity)
            ffnn_hidden_entity.append(hidden_entity)
        # logits_entity = ffnn_hidden_entity[-1]  # B * SENT_LEN * NUM_LABELS
        logits = ffnn_hidden_entity[-1]  # B * SENT_LEN * NUM_LABELS

        # if self.dual_classifier:
        #     ffnn_hidden_trigger = []
        #     hidden_trigger = spans_embedding
        #     for layer in self.trigger_classifier:
        #         hidden_trigger = layer(hidden_trigger)
        #         ffnn_hidden_trigger.append(hidden_trigger)
        #     logits_trigger = ffnn_hidden_trigger[-1]
        #     logits = torch.cat((logits_entity, logits_trigger), dim=2)
        # else:
            # logits = logits_entity

        if spans_ner_label is not None:
            loss_fct = CrossEntropyLoss(reduction='sum')
            if attention_mask is not None:
                active_loss = spans_mask.view(-1) == 1
                active_logits = logits.view(-1, logits.shape[-1])
                active_labels = torch.where(
                    active_loss, spans_ner_label.view(-1), torch.tensor(loss_fct.ignore_index).type_as(spans_ner_label)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, logits.shape[-1]), spans_ner_label.view(-1))
            return loss, logits, spans_embedding
        else:
            return logits, spans_embedding, spans_embedding
        

class EntityModel():

    def __init__(self, args, num_entity_labels, tokenizer, evaluation=False):
        super().__init__()

        bert_model_name = args.model
        # vocab_name = bert_model_name
        
        if args.bert_model_dir is not None:
            bert_model_name = str(args.bert_model_dir) + '/'
            # vocab_name = bert_model_name + 'vocab.txt'
            # vocab_name = bert_model_name
            logger.info('Loading BERT model from {}'.format(bert_model_name))

        self.tokenizer = tokenizer
        self.bert_model = BertForEntity.from_pretrained(
            bert_model_name, 
            num_entity_labels=num_entity_labels, 
            max_span_length=args.max_span_length_entity,
            sent_pos_embedding_dim=args.num_segment_doc,
            add_relative_position=args.add_relative_position,
            add_special_tokens=args.add_special_tokens
        )
        if args.add_special_tokens:
            self.bert_model.resize_token_embeddings(len(self.tokenizer))

        self._model_device = 'cpu'
        self.move_model_to_cuda()

        self.cnt = 0

    def move_model_to_cuda(self):
        if not torch.cuda.is_available():
            logger.error('No CUDA found!')
            return
        logger.info('Moving to CUDA...')
        self._model_device = 'cuda'
        self.bert_model.cuda()
        logger.info('# GPUs = %d'%(torch.cuda.device_count()))
        if torch.cuda.device_count() > 1:
            self.bert_model = torch.nn.DataParallel(self.bert_model)
            logger.info(f"Multi-GPU mode with {torch.cuda.device_count()} GPUs")

    def _get_input_tensors(self, tokens, spans, spans_ner_label, sent_header_indices):

        start2idx = []
        end2idx = []
        
        bert_tokens = []
        bert_tokens.append(self.tokenizer.cls_token)
        for token in tokens:
            start2idx.append(len(bert_tokens))
            # sub_tokens = self.tokenizer.tokenize(token.lower())
            sub_tokens = self.tokenizer.tokenize(token)
            bert_tokens += sub_tokens
            end2idx.append(len(bert_tokens)-1)
        bert_tokens.append(self.tokenizer.sep_token)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        tokens_tensor = torch.tensor([indexed_tokens])

        # Convert token-level span to subtoken-level span
        bert_spans = [[start2idx[span[0]], end2idx[span[1]], span[2]] for span in spans] 
        bert_spans_tensor = torch.tensor([bert_spans])
        spans_ner_label_tensor = torch.tensor([spans_ner_label])

        bert_sent_header_indices = [start2idx[idx] for idx in sent_header_indices]
        bert_sent_header_indices_tensor = torch.tensor([bert_sent_header_indices])

        if self.cnt < 20:
            print()
            print(bert_tokens)
            print(indexed_tokens)
            print(bert_spans)
            print(spans_ner_label_tensor)
        self.cnt += 1
        # if self.cnt == 20:
        #     exit(0)

        return tokens_tensor, bert_spans_tensor, spans_ner_label_tensor, bert_sent_header_indices_tensor
    

    def _get_input_tensors_batch(self, samples_list, training=True):
        tokens_tensor_list = []
        bert_spans_tensor_list = []
        spans_ner_label_tensor_list = []
        sent_header_indices_tensor_list = []
        sent_pos_in_doc_tensor_list = []
        sentence_length = []

        max_tokens = 0
        max_spans = 0
        for sample in samples_list:

            tokens = sample['tokens']
            spans = sample['spans']
            spans_ner_label = sample['spans_label']
            sent_header_indices = sample['sent_header_idx']

            tokens_tensor, bert_spans_tensor, spans_ner_label_tensor, sent_header_indices_tensor = self._get_input_tensors(tokens, spans, spans_ner_label, sent_header_indices)
            tokens_tensor_list.append(tokens_tensor)
            bert_spans_tensor_list.append(bert_spans_tensor)
            spans_ner_label_tensor_list.append(spans_ner_label_tensor)
            sent_header_indices_tensor_list.append(sent_header_indices_tensor)

            sent_pos_in_doc_tensor_list.append(torch.tensor([sample['sent_pos_in_doc']]))

            assert (bert_spans_tensor.shape[1] == spans_ner_label_tensor.shape[1])

            if (tokens_tensor.shape[1] > max_tokens):
                max_tokens = tokens_tensor.shape[1]
            if (bert_spans_tensor.shape[1] > max_spans):
                max_spans = bert_spans_tensor.shape[1]

            sentence_length.append(sample['sent_length'])
            
        sentence_length = torch.Tensor(sentence_length)

        # apply padding and concatenate tensors
        final_tokens_tensor = None
        final_attention_mask = None
        final_bert_spans_tensor = None
        final_spans_ner_label_tensor = None
        final_spans_mask_tensor = None
        for tokens_tensor, bert_spans_tensor, spans_ner_label_tensor, sent_header_indices_tensor, sent_pos_in_doc_tensor in zip(tokens_tensor_list, bert_spans_tensor_list, spans_ner_label_tensor_list, sent_header_indices_tensor_list, sent_pos_in_doc_tensor_list):
            # padding for tokens
            num_tokens = tokens_tensor.shape[1]
            tokens_pad_length = max_tokens - num_tokens
            attention_tensor = torch.full([1,num_tokens], 1, dtype=torch.long)
            if tokens_pad_length>0:
                pad = torch.full([1,tokens_pad_length], self.tokenizer.pad_token_id, dtype=torch.long)
                tokens_tensor = torch.cat((tokens_tensor, pad), dim=1)
                attention_pad = torch.full([1,tokens_pad_length], 0, dtype=torch.long)
                attention_tensor = torch.cat((attention_tensor, attention_pad), dim=1)

            # padding for spans
            num_spans = bert_spans_tensor.shape[1]
            spans_pad_length = max_spans - num_spans
            spans_mask_tensor = torch.full([1,num_spans], 1, dtype=torch.long)
            if spans_pad_length > 0:
                pad = torch.full([1, spans_pad_length, bert_spans_tensor.shape[2]], 0, dtype=torch.long)
                bert_spans_tensor = torch.cat((bert_spans_tensor, pad), dim=1)
                mask_pad = torch.full([1, spans_pad_length], 0, dtype=torch.long)
                spans_mask_tensor = torch.cat((spans_mask_tensor, mask_pad), dim=1)
                spans_ner_label_tensor = torch.cat((spans_ner_label_tensor, mask_pad), dim=1)
                sent_header_indices_tensor = torch.cat((sent_header_indices_tensor, mask_pad), dim=1)
                sent_pos_in_doc_tensor = torch.cat((sent_pos_in_doc_tensor, mask_pad), dim=1)

            # update final outputs
            if final_tokens_tensor is None:
                final_tokens_tensor = tokens_tensor
                final_attention_mask = attention_tensor
                final_bert_spans_tensor = bert_spans_tensor
                final_spans_ner_label_tensor = spans_ner_label_tensor
                final_sent_header_indices_tensor = sent_header_indices_tensor
                final_sent_pos_in_doc_tensor = sent_pos_in_doc_tensor
                final_spans_mask_tensor = spans_mask_tensor
            else:
                final_tokens_tensor = torch.cat((final_tokens_tensor, tokens_tensor), dim=0)
                final_attention_mask = torch.cat((final_attention_mask, attention_tensor), dim=0)
                final_bert_spans_tensor = torch.cat((final_bert_spans_tensor, bert_spans_tensor), dim=0)
                final_spans_ner_label_tensor = torch.cat((final_spans_ner_label_tensor, spans_ner_label_tensor), dim=0)
                final_sent_header_indices_tensor = torch.cat((final_sent_header_indices_tensor, sent_header_indices_tensor), dim=0)
                final_sent_pos_in_doc_tensor = torch.cat((final_sent_pos_in_doc_tensor, sent_pos_in_doc_tensor), dim=0)
                final_spans_mask_tensor = torch.cat((final_spans_mask_tensor, spans_mask_tensor), dim=0)

        return final_tokens_tensor, final_attention_mask, final_bert_spans_tensor, final_spans_mask_tensor, final_spans_ner_label_tensor, final_sent_header_indices_tensor, final_sent_pos_in_doc_tensor, sentence_length


    def run_batch(self, samples_list, try_cuda=True, training=True):

        # convert samples to input tensors
        tokens_tensor, attention_mask_tensor, bert_spans_tensor, spans_mask_tensor, spans_ner_label_tensor, sent_header_indices_tensor, sent_pos_in_doc_tensor, sentence_length = self._get_input_tensors_batch(samples_list, training)

        output_dict = {
            'ner_loss': 0,
        }

        if training:
            self.bert_model.train()
            ner_loss, ner_logits, spans_embedding = self.bert_model(
                input_ids = tokens_tensor.to(self._model_device),
                spans = bert_spans_tensor.to(self._model_device),
                spans_mask = spans_mask_tensor.to(self._model_device),
                spans_ner_label = spans_ner_label_tensor.to(self._model_device),
                attention_mask = attention_mask_tensor.to(self._model_device),
                sent_header_indices = sent_header_indices_tensor.to(self._model_device),
                sent_pos_in_doc = sent_pos_in_doc_tensor.to(self._model_device)
            )
            output_dict['ner_loss'] = ner_loss.sum()
            output_dict['ner_llh'] = F.log_softmax(ner_logits, dim=-1)
        else:
            self.bert_model.eval()
            with torch.no_grad():
                ner_logits, spans_embedding, last_hidden = self.bert_model(
                    input_ids = tokens_tensor.to(self._model_device),
                    spans = bert_spans_tensor.to(self._model_device),
                    spans_mask = spans_mask_tensor.to(self._model_device),
                    spans_ner_label = None,
                    attention_mask = attention_mask_tensor.to(self._model_device),
                    sent_header_indices = sent_header_indices_tensor.to(self._model_device),
                    sent_pos_in_doc = sent_pos_in_doc_tensor.to(self._model_device)
                )
            _, predicted_label = ner_logits.max(2)
            predicted_label = predicted_label.cpu().numpy()
            last_hidden = last_hidden.cpu().numpy()
            
            predicted = []
            pred_prob = []
            hidden = []
            for i, sample in enumerate(samples_list):
                ner = []
                prob = []
                lh = []
                for j in range(len(sample['spans'])):
                    ner.append(predicted_label[i][j])
                    # prob.append(F.softmax(ner_logits[i][j], dim=-1).cpu().numpy())
                    prob.append(ner_logits[i][j].cpu().numpy())
                    lh.append(last_hidden[i][j])
                predicted.append(ner)
                pred_prob.append(prob)
                hidden.append(lh)
            output_dict['pred_ner'] = predicted
            output_dict['ner_probs'] = pred_prob
            output_dict['ner_last_hidden'] = hidden

        return output_dict
