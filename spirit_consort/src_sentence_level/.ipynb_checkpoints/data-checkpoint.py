import torch
from collections import namedtuple
from transformers import BertTokenizer, BertModel
from collections import Counter
import ast, math, random, json
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import os, sys

batch_fields = ['text_ids', 'attention_mask_text', 'labels', 'PMCID', 'section', 'index', 'loss_mask', 'contrastive_flag']
Batch = namedtuple('Batch', field_names=batch_fields,
                   defaults=[None] * len(batch_fields))

instance_fields = ['text', 'text_ids', 'attention_mask_text',
                   'labels', 't_labels', 'section', 'PMCID', 'index', 'loss_mask']
Instance = namedtuple('Instance', field_names=instance_fields,
                      defaults=[None] * len(instance_fields))


# pd.set_option('display.max_columns', 10)
# pd.set_option('display.max_rows', 1000)
# pd.set_option('display.max_colwidth', -1)


def process_contextual_for_bert(tokenizer, data, max_length=128):
    instances = []
    index = 0
    for i in data:
        text_ids = tokenizer.encode(i[0],
                                    add_special_tokens=False,
                                    truncation=True,
                                    max_length=max_length)
        # pad_num = max_length - len(text_ids)
        # text_ids = [2] + text_ids + [3]

        text_ids_before = tokenizer.encode(i[2],
                                           add_special_tokens=False,
                                           truncation=True,
                                           max_length=max_length)

        text_ids_after = tokenizer.encode(i[3],
                                          add_special_tokens=False,
                                          truncation=True,
                                          max_length=max_length)

        text_ids_all = [2] + text_ids_before + [3] + text_ids + [3] + text_ids_after + [3]
        pad_num = 3 * 128 + 4 - len(text_ids_all)
        attn_mask = [1] * len(text_ids_all) + [0] * pad_num
        text_ids_all = text_ids_all + [0] * pad_num

        labels = list(i[1])
        instance = Instance(
            index=index,
            text_ids=text_ids_all,
            attention_mask_text=attn_mask,
            labels=labels,
            section=i[-1],
            t_labels=i[-2],
            PMCID=i[-3]
        )
        instances.append(instance)
        index += 1
    return instances


def process_single_for_bert(tokenizer, data, label_convert=None, max_length=128):
    instances = []
    if label_convert:
        labels_tokens = [tokenizer.encode(i, add_special_tokens=False) for i in label_convert]
    for i in data:
        text_ids = tokenizer.encode(i[0],
                                    add_special_tokens=False,
                                    truncation=True,
                                    max_length=max_length)
        if label_convert:
            text_ids = labels_tokens + text_ids
        pad_num = max_length - len(text_ids)
        attn_mask = [1] * len(text_ids) + [0] * pad_num
        text_ids = text_ids + [0] * pad_num
        labels = list(i[1])
        instance = Instance(
            text_ids=text_ids,
            attention_mask_text=attn_mask,
            labels=labels,
            section=i[-1],
            t_labels=i[-2],
            PMCID=i[-3]
        )
        instances.append(instance)
    return instances


def collate_fn(batch, gpu=True):
    batch_text_idxs = []
    batch_attention_masks_text = []
    batch_labels = []
    batch_index = []
    batch_pmcid = []
    batch_section = []
    contrastive_flag = False
    # batch_section_idxs, batch_attention_masks_section = [], []
    for inst in batch:
        batch_text_idxs.append(inst.text_ids)
        batch_attention_masks_text.append(inst.attention_mask_text)
        batch_labels.append(inst.labels)
        batch_index.append(inst.index)
        batch_pmcid.append(inst.PMCID)
        batch_section.append(inst.section)
        if inst.t_labels != []:
            contrastive_flag = True
    if gpu:
        batch_text_idxs = torch.cuda.LongTensor(batch_text_idxs)
        batch_attention_masks_text = torch.cuda.FloatTensor(batch_attention_masks_text)
        batch_labels = torch.cuda.FloatTensor(batch_labels)
    else:
        batch_text_idxs = torch.LongTensor(batch_text_idxs)
        batch_attention_masks_text = torch.FloatTensor(batch_attention_masks_text)
        batch_labels = torch.FloatTensor(batch_labels)
    return Batch(
        text_ids=batch_text_idxs,
        attention_mask_text=batch_attention_masks_text,
        index=batch_index,
        section=batch_section,
        PMCID=batch_pmcid,
        contrastive_flag=contrastive_flag,
        labels=batch_labels,
    )


def load_file(path):
    file = pd.read_csv(path, header=0)
    file['ChecklistItem'] = file['ChecklistItem'].apply(ast.literal_eval)
    file['SectionHeaders'] = file['SectionHeaders'].apply(ast.literal_eval)
    # create a new column to state max sentence number of each paper
    dict_n_sen = dict()
    file['SentenceID'] = file['SentenceID'].apply(int)
    for id, article in file.groupby('PMCID'):
        dict_n_sen[id] = max(article['SentenceID'])
    file['MaxSen'] = file.PMCID.replace(to_replace=dict_n_sen)
    return file


def load_labels(data_folder):
    with open(data_folder + 'labels.txt', 'r') as f:
        f = f.read().split('\n')
    return f


def dataset_construct(data, label_convert, tokenizer):
    extracted_data = []
    for index, article in data.groupby('PMCID'):
        for i, sentence in article.iterrows():
            max_sentence = max(article['SentenceID'])
            if sentence['SentenceID'] == 0:
                previous_sent = ''
                if article[article['SentenceID'] == sentence['SentenceID'] + 1].shape[0] != 0:
                    next_s = article[article['SentenceID'] == sentence['SentenceID'] + 1].iloc[0]
                    next_sentence = ' '.join(next_s['SectionHeaders']) + ' ' + next_s['SentenceNoMarkers']
                else:
                    next_sentence = ''
            elif sentence['SentenceID'] == max_sentence:
                if article[article['SentenceID'] == sentence['SentenceID'] - 1].shape[0] != 0:
                    previous_s = article[article['SentenceID'] == sentence['SentenceID'] - 1].iloc[0]
                    previous_sent = ' '.join(previous_s['SectionHeaders']) + ' ' + previous_s['SentenceNoMarkers']
                else:
                    previous_sent = ''
                next_sentence = ''
            else:
                if article[article['SentenceID'] == sentence['SentenceID'] - 1].shape[0] != 0:
                    previous_s = article[article['SentenceID'] == sentence['SentenceID'] - 1].iloc[0]
                    previous_sent = ' '.join(previous_s['SectionHeaders']) + ' ' + previous_s['SentenceNoMarkers']
                else:
                    previous_sent = ''
                if article[article['SentenceID'] == sentence['SentenceID'] + 1].shape[0] != 0:
                    next_s = article[article['SentenceID'] == sentence['SentenceID'] + 1].iloc[0]
                    next_sentence = ' '.join(next_s['SectionHeaders']) + ' ' + next_s['SentenceNoMarkers']

                else:
                    next_sentence = ''

            text = ' '.join(sentence['SectionHeaders']) + ' ' + sentence['SentenceNoMarkers']
            extracted_data.append([text, label_convert.transform([sentence['ChecklistItem']])[0], previous_sent,
                                   next_sentence, sentence.PMCID, sentence['ChecklistItem'], str(sentence['SectionHeaders'])])
    processed_data = process_contextual_for_bert(tokenizer, extracted_data)
    return processed_data


def dataset_construct_single(data, label_convert, tokenizer):
    extracted_data = []
    for i, row in data.iterrows():
        text = ' '.join(row['SectionHeaders']) + ' ' + row['SentenceNoMarkers']
        extracted_data.append(
            [text, label_convert.transform([row['ChecklistItem']])[0], row.PMCID, row['ChecklistItem'], str(row['SectionHeaders'])])
    processed_data = process_single_for_bert(tokenizer, extracted_data)
    return processed_data


def contextual_load(config, mode='train'):
    file = load_file(config.data_path)
    if mode == 'train':
        train = file[file['Split']=='train']
        train = train[train['IsSectionHeader'] != 1]
        test = file[file['Split']=='test']
        test = test[test['IsSectionHeader'] != 1]
    valid = file[file['Split']=='valid']
    valid = valid[valid['IsSectionHeader'] != 1]

    # load labels
    labels = set([j for i in file.ChecklistItem for j in i])
    label_convert = MultiLabelBinarizer()
    label_convert.fit([labels])

    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name, do_lower_case=True)
    if mode == 'train':
        train = dataset_construct(train, label_convert, tokenizer)
        print("INFO: TRAIN LOADED.")
        test = dataset_construct(test, label_convert, tokenizer)
        print("INFO: TEST LOADED.")
    valid = dataset_construct(valid, label_convert, tokenizer)
    print("INFO: VALID LOADED.")

    labels = list(label_convert.classes_)
    if mode == 'train':
        return train, valid, test, labels
    else:
        return valid, labels


def single_load(config, mode='train'):
    if mode == 'train':
        train = load_file(config.data_folder + config.train_file)
        train = train[train['IsSectionHeader'] != 'Yes']
        valid = load_file(config.data_folder + config.valid_file)
        valid = valid[valid['IsSectionHeader'] != 'Yes']
    test = load_file(config.data_folder + config.test_file)
    test = test[test['IsSectionHeader'] != 'Yes']
    # load labels
    labels = load_labels(config.data_folder)
    label_convert = MultiLabelBinarizer()
    label_convert.fit([labels])

    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name, do_lower_case=True)
    if mode == 'train':
        train = dataset_construct_single(train, label_convert, tokenizer)
        print("INFO: TRAIN LOADED.")
        valid = dataset_construct_single(valid, label_convert, tokenizer)
        print("INFO: VALID LOADED.")
    test = dataset_construct_single(test, label_convert, tokenizer)
    print("INFO: TEST LOADED.")

    labels = list(label_convert.classes_)
    if mode == 'train':
        return train, valid, test, labels, tokenizer
    else:
        return valid, labels, tokenizer
