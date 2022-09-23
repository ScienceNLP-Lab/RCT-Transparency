import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader

import os
import pandas as pd
from collections import namedtuple
import numpy as np
import tqdm
from config import Config
from sentence_segmentation import segment
from bert_model import BERT


batch_fields = ['text_ids', 'attention_mask_text']
Batch = namedtuple('Batch', field_names=batch_fields,
                   defaults=[None] * len(batch_fields))

def colloate_fn(batch):
    batch_text_idxs = []
    batch_attention_masks_text = []
    for i in batch:
        batch_text_idxs.append(i[0])
        batch_attention_masks_text.append(i[1])
    batch_text_idxs = torch.cuda.LongTensor(batch_text_idxs)
    batch_attention_masks_text = torch.cuda.FloatTensor(batch_attention_masks_text)
    return Batch(text_ids=batch_text_idxs, attention_mask_text=batch_attention_masks_text)

def data_process(folder_path, config):
    list_file = []
    for (dirpath, dirnames, filenames) in os.walk(folder_path):
        for filename in filenames:
            if '.txt' in filename:
                f = open(dirpath + '/' + filename)
                f = f.readline()
                if len(f) != 0:
                    f = segment(f)
                    for s in f:
                        list_file.append([dirpath.split('/')[-1], filename, s['sentence'], s['span']])
    file = pd.DataFrame(list_file, columns=['folder', 'file', 'sentence', 'span'])
    print("INFO: DATA PARSED")

    sentences = []
    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name, do_lower_case=True)
    for i, row in file.iterrows():
        text_ids = tokenizer.encode(row['sentence'],
                                    add_special_tokens=True,
                                    truncation=True,
                                    max_length=128)
        pad_num = 128 - len(text_ids)
        attn_mask = [1] * len(text_ids) + [0] * pad_num
        text_ids = text_ids + [0] * pad_num
        sentences.append([text_ids, attn_mask])
    return sentences, file

def main(state_path, model_path, folder_path, saved_path):
    print("INFO: MODEL PATH:", os.path.join(model_path, "method_consort_model.mdl"))
    map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = torch.load(os.path.join(state_path, "method_consort_model.mdl"), map_location=map_location)
    config = state['config']
    if type(config) is dict:
        config = Config.from_dict(config)
    setattr(config, 'bert_model_name', model_path)

    model = BERT(config)
    model.load_bert(config.bert_model_name)
    model.load_state_dict(state['model'])
    model.cuda(0)

    sentences, file = data_process(folder_path, config)
    print("INFO: DATA LOADED")

    batch_num = len(sentences) // 4 + (len(sentences) % 4)
    all_result = []
    progress = tqdm.tqdm(total=batch_num, ncols=75)

    for batch in DataLoader(sentences, batch_size=4, shuffle=False,
                            collate_fn=colloate_fn):
        model.eval()
        result = model(batch)
        result = result.cpu().data
        result = np.where(result.numpy() > 0.5, 1, 0)
        all_result.extend(result.tolist())
        progress.update(1)
    progress.close()

    column = ['10', '11a', '11b', '12a', '12b', '3a', '3b', '4a', '4b', '5', '6a',
              '6b', '7a', '7b', '8a', '8b', '9']
    labels = []
    for inst in all_result:
        label = []
        for j in range(len(inst)):
            if inst[j] == 1:
                label.append(column[j])
        if len(label) == 0:
            label.append('0')
        labels.append(label)
    file['label'] = labels
    file.to_csv(saved_path, index=False)
    print("INFO: FILE SAVED")

if __name__ == '__main__':
    # path to the model
    state_path = '/efs/lanj3/20220830_135624/20220830_135624/'
    model_path = '/efs/lanj3/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/'
    # path to the text folder
    folder_path = '/efs/lanj3/comparison_set_individual_files/full_texts'
    # path to the file to be saved
    saved_path = '/efs/lanj3/prediction_covid.csv'
    main(state_path, model_path, folder_path, saved_path)
