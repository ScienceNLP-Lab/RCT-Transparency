import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F
from transformers import BertConfig


def token_lens_to_idxs(token_lens):
    """Map token lengths to a word piece index matrix (for torch.gather) and a
    mask tensor.
    For example (only show a sequence instead of a batch):

    token lengths: [1,1,1,3,1]
    =>
    indices: [[0,0,0], [1,0,0], [2,0,0], [3,4,5], [6,0,0]]
    masks: [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.33, 0.33, 0.33], [1.0, 0.0, 0.0]]

    Next, we use torch.gather() to select vectors of word pieces for each token,
    and average them as follows (incomplete code):

    outputs = torch.gather(bert_outputs, 1, indices) * masks
    outputs = bert_outputs.view(batch_size, seq_len, -1, self.bert_dim)
    outputs = bert_outputs.sum(2)

    :param token_lens (list): token lengths.
    :return: a index matrix and a mask tensor.
    """
    max_token_num = max([len(x) for x in token_lens])
    max_token_len = max([max(x) for x in token_lens])
    idxs, masks = [], []
    for seq_token_lens in token_lens:
        seq_idxs, seq_masks = [], []
        offset = 0
        for token_len in seq_token_lens:
            seq_idxs.extend([i + offset for i in range(token_len)]
                            + [-1] * (max_token_len - token_len))
            seq_masks.extend([1.0 / token_len] * token_len
                             + [0.0] * (max_token_len - token_len))
            offset += token_len
        seq_idxs.extend([-1] * max_token_len * (max_token_num - len(seq_token_lens)))
        seq_masks.extend([0.0] * max_token_len * (max_token_num - len(seq_token_lens)))
        idxs.append(seq_idxs)
        masks.append(seq_masks)
    return idxs, masks, max_token_num, max_token_len


class BERT(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        bert_config = BertConfig.from_pretrained(config.bert_model_name, output_all_encoded_layers=False)
        # self.bert = BertModel(bert_config)
        self.bert_dim = bert_config.hidden_size
        self.drop = nn.Dropout(p=config.bert_dropout)
        self.add_section = config.section_emb
        self.add_position = config.position_emb
        self.section = nn.Linear(self.bert_dim, config.section_dim)
        self.rltv = config.rltv
        self.section_avg_sep = config.section_avg_sep
        if config.mode!="contextual" and config.augmentation_mode == 0:
            self.abs_pos_emb = nn.Embedding(num_embeddings=config.sen_num, embedding_dim=config.sent_dim)
            self.rltv_pos_emb = nn.Embedding(num_embeddings=config.rltv_bins_num, embedding_dim=config.sent_dim)
        # self.labels_num = len(['10', '11a', '11b', '12a', '12b', '3a', '3b', '4a', '4b', '5', '6a',
        #           '6b', '7a', '7b', '8a', '8b', '9'])
        # self.fc_guidence = nn.Linear(self.bert_dim, 100)
        if config.position_emb:
            self.fc1 = nn.Linear(self.bert_dim + config.sent_dim, num_labels)
        elif config.section_emb:
            self.fc1 = nn.Linear(self.bert_dim + config.section_dim, num_labels)
        else:
            self.fc1 = nn.Linear(self.bert_dim + config.section_dim, num_labels)
        # self.fc2 = nn.Linear(50, 35)
        self.sim = nn.CosineSimilarity(dim=2)
        self.sig = nn.Sigmoid()
        self.num_labels = num_labels
        self.target = config.target

    def load_bert(self, name, cache_dir=None):
        """Load the pre-trained BERT model (used in training phrase)
        :param name (str): pre-trained BERT model name
        :param cache_dir (str): path to the BERT cache directory
        """
        print('INFO: Loading pre-trained BERT model {}'.format(name))
        self.bert = BertModel.from_pretrained(name)

    def forward(self, batch, guidance=None):
        output = self.bert(batch.text_ids, attention_mask=batch.attention_mask_text)
        if self.add_position:
            if self.rltv == 1:
                pos_emb = self.rltv_pos_emb(batch.rltv_id)
            elif self.rltv == 0:
                pos_emb = self.abs_pos_emb(batch.sent_id-1)
        if self.add_section:
            section_output = self.bert(batch.section_ids, attention_mask=batch.attention_mask_section)
            # average all pieces for multi-piece words
            if self.section_avg_sep == "avg":
                batch_size = section_output[0].shape[0]
                section_num = [len(n) for n in batch.sec_token_len]
                idxs, masks, token_num, token_len = token_lens_to_idxs(batch.sec_token_len)
                idxs = batch.section_ids.new(idxs).unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + 1
                masks = section_output[0].new(masks).unsqueeze(-1)
                bert_outputs = torch.gather(section_output[0], 1, idxs) * masks
                section_output = bert_outputs.view(batch_size, token_num, token_len, self.bert_dim)

                bert_outputs = bert_outputs.view(batch_size, token_num, token_len, self.bert_dim)
                section_output = bert_outputs.sum(2)
                section_emb = []

                for section in range(batch_size):
                    section_emb.append(torch.mean(section_output[section][:section_num[section]], 0))
                section_output = torch.stack(section_emb)

            elif self.section_avg_sep == "sep":
                section_output = section_output[1]
                

        # output = torch.cat(output[1], guidence, dim=0)
        if self.target == "[CLS]":
            out = self.drop(output[1])
            if self.add_section:
                sec_out = self.section(section_output)
                out = torch.cat((out, sec_out), 1)
            if self.add_position: 
                out = torch.cat((out, pos_emb), 1)
            out = self.fc1(out)
            out = self.sig(out)
        elif self.target == "[SEP]":
            out = self.drop(output[0])
            out = self.fc1(out)
            out = self.sig(out)
        
        return out

