import torch
import torch.nn as nn
from transformers import BertModel
from transformers import BertConfig

class BERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        bert_config = BertConfig.from_pretrained(config.bert_model_name, output_all_encoded_layers=False)
        self.bert_dim = bert_config.hidden_size
        self.drop = nn.Dropout(p=config.bert_dropout)
        self.fc1 = nn.Linear(self.bert_dim, 34)
        self.sig = nn.Sigmoid()

    def load_bert(self, name, cache_dir=None):
        """Load the pre-trained BERT model (used in training phrase)
        :param name (str): pre-trained BERT model name
        :param cache_dir (str): path to the BERT cache directory
        """
        print('INFO: Loading pre-trained BERT model {}'.format(name))
        self.bert = BertModel.from_pretrained(name)

    def forward(self, batch):
        output = self.bert(batch.text_ids, attention_mask=batch.attention_mask_text)
        out = self.drop(output[1])
        out = self.fc1(out)
        out = self.sig(out)
        return out

