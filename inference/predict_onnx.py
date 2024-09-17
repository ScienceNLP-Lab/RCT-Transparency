import ast

import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader

import nltk, requests
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from sklearn.preprocessing import MultiLabelBinarizer

import os, tqdm, datetime, re
import pandas as pd
from collections import namedtuple
import numpy as np
import onnxruntime as ort

nltk.download('punkt')
porter = PorterStemmer()
lancaster = LancasterStemmer()

batch_fields = ['text_ids', 'attention_mask_text']
Batch = namedtuple('Batch', field_names=batch_fields,
                   defaults=[None] * len(batch_fields))
instance_fields = ['text_ids', 'attention_mask_text']
Instance = namedtuple('Instance', field_names=instance_fields,
                      defaults=[None] * len(instance_fields))

def colloate_fn(batch, gpu=False):
    batch_text_idxs = []
    batch_attention_masks_text = []
    for i in batch:
        batch_text_idxs.append(i[0])
        batch_attention_masks_text.append(i[1])
    # if gpu:
    #     batch_text_idxs = torch.cuda.LongTensor(batch_text_idxs)
    #     batch_attention_masks_text = torch.cuda.FloatTensor(batch_attention_masks_text)
    # else:
    #     batch_text_idxs = torch.LongTensor(batch_text_idxs)
    #     batch_attention_masks_text = torch.FloatTensor(batch_attention_masks_text)
    return Batch(text_ids=batch_text_idxs, attention_mask_text=batch_attention_masks_text)


def process_contextual_for_bert(tokenizer, data, max_length=128):
    instances = []
    for i in data:
        text_ids = tokenizer.encode(i[3],
                                    add_special_tokens=False,
                                    truncation=True,
                                    max_length=max_length)
        text_ids_before = tokenizer.encode(i[4],
                                           add_special_tokens=False,
                                           truncation=True,
                                           max_length=max_length)
        text_ids_after = tokenizer.encode(i[5],
                                          add_special_tokens=False,
                                          truncation=True,
                                          max_length=max_length)

        text_ids_all = [2] + text_ids_before + [3] + text_ids + [3] + text_ids_after + [3]
        pad_num = 3 * 128 + 4 - len(text_ids_all)
        attn_mask = [1] * len(text_ids_all) + [0] * pad_num
        text_ids_all = text_ids_all + [0] * pad_num

        instance = Instance(text_ids=text_ids_all, attention_mask_text=attn_mask)
        instances.append(instance)
    return instances


def unique(section_header):
    headers = []
    for i in section_header:
        if i not in headers:
            headers.append(i)
    return headers


def contextual_load(path, tokenizer_path):
    data = []
    file = pd.read_csv(path, header=0)
    file['sentence_id'] = file['sentence_id'].apply(lambda x: int(x[1:]))
    file['section'] = file['section'].apply(ast.literal_eval)
    file['section'] = file.apply(lambda row: list([row['section.1']] + row.section), axis=1)
    file['section'] = file['section'].map(unique)

    for pmcid, article in file.groupby('PMCID'):
        max_sentence = max(article['sentence_id'])
        for sent_id, sentence in article.iterrows():
            if sentence['sentence_id'] == 1:
                previous_sent = ''
                next_s = article[article['sentence_id'] == sentence['sentence_id']+1].iloc[0]
                next_sentence = ' '.join(next_s['section']) + ' ' + next_s['text']
            elif sentence['sentence_id'] == max_sentence:
                previous_s = article[article['sentence_id'] == sentence['sentence_id']-1].iloc[0]
                previous_sent = ' '.join(previous_s['section']) + ' ' + previous_s['text']
                next_sentence = ''
            else:
                previous_s = article[article['sentence_id'] == sentence['sentence_id']-1].iloc[0]
                previous_sent = ' '.join(previous_s['section']) + ' ' + previous_s['text']
                next_s = article[article['sentence_id'] == sentence['sentence_id']+1].iloc[0]
                next_sentence = ' '.join(next_s['section']) + ' ' + next_s['text']
            data.append([pmcid, sentence['section.1'], sentence['sentence_id'], ' '.join(sentence['section'])
                         + ' '+sentence['text'], previous_sent, next_sentence])
    file = pd.DataFrame(data, columns=['PMCID', 'section', 'sentence_id', 'sentence', 'before', 'after'])
    print("INFO: DATA PARSED")

    tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=True)
    data = process_contextual_for_bert(tokenizer, data)
    return data, file


# 3a, 3b
def stem_sentence(sentence):
    tokenized_words = word_tokenize(sentence)
    tokenized_sentence = []
    for word in tokenized_words:
        tokenized_sentence.append(porter.stem(word))
    tokenized_sentence = " ".join(tokenized_sentence)
    return tokenized_sentence


phrases_3b = [stem_sentence(i) for i in ["no longer feasible", "decision was taken",
                                         "decision was made", "committee agreed"]]
phrases_6b = [stem_sentence(i) for i in ["original trial protocol", "trial because of",
                                         "early termination", "the trial because", "the original trial"]]


def check_content(sentence, phrases):
    stemmed_sentence = stem_sentence(sentence)
    for phrase in phrases:
        if phrase in stemmed_sentence:
            return True
    return False


def check_title(l1):
    if l1.lower() in ["title", "titles"]:
        return True
    return False


def check_1a(text):
    words = word_tokenize(text)
    for w in words:
        w = w.split("-")
        for i in w:
            if porter.stem(i.lower()) in ["random", "randomis"]:
                return True
    return False


def find_1b_candidates(str1):
    if str1.lower() in ['summary', 'abstract']:
        return True
    return False


def check_1b():
    response = requests.get(
        "https://lhncbc.nlm.nih.gov/ii/areas/structured-abstracts/downloads/Structured-Abstracts-Labels-102615.txt")
    data = response.text
    structured_abstract_items = []

    for i in data.split("\n"):
        structured_abstract_items.append(i.split("|")[0])
    structured_abstract_items.remove("")


def structure():
    response = requests.get(
        "https://lhncbc.nlm.nih.gov/ii/areas/structured-abstracts/downloads/Structured-Abstracts-Labels-102615.txt")
    data = response.text
    structured_abstract_items = []

    for i in data.split("\n"):
        structured_abstract_items.append(i.split("|")[0])
    structured_abstract_items.remove("")
    return structured_abstract_items


def check_start(string, sample):
    string = string.lower()
    sample = sample.lower()
    if (sample in string):
        y = "^" + sample
        x = re.search(y, string)
        if x:
            return True
        else:
            return False
    else:
        return False


def check_1b_by_map(str1, structured_abstract_items):
    for i in structured_abstract_items:
        if check_start(str1.lower(), i):
            return True
    return False


def merge(ml_ppredict, rule_based, article_level):
    merged = [] + rule_based +ml_ppredict
    merged = [i for i in merged if i != '0']
    if type(article_level) == list:
        merged = merged + article_level
    if len(merged) == 0:
        merged = ['0']
    return merged


def main_consort(model_path, tokenizer_name, csv_path, saved_path, column):
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    ort_sess = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])

    sentences, file = contextual_load(csv_path, tokenizer_name)
    print("INFO: DATA LOADED")

    # for 1a, 1b, 3b, 6b
    rule_based_prediction = []
    for i, row in file.iterrows():
        pred_instance = []
        if check_content(row['sentence'], phrases_3b):
            pred_instance.append('3b')
        if check_content(row['sentence'], phrases_6b):
            pred_instance.append('6b')
        rule_based_prediction.append([row['PMCID'], row['sentence_id'], pred_instance])
    rule_based_prediction = pd.DataFrame(rule_based_prediction, columns=['PMCID', 'sentence_id', '3b_6b'])

    # for 1a and 1b
    article_prediction = []
    abs_stru = structure()
    for pmcid, article in file.groupby('PMCID'):
        pred_a, pred_b = False, False
        title_flag = article['section'].apply(check_title)
        for i, row in article[title_flag].iterrows():
            if not pred_a:
                pred_a = check_1a(row['sentence'])
                if pred_a:
                    article_prediction.append([pmcid, row['sentence_id'], '1a'])
            else:
                break
        abstract_flag = article["section"].apply(find_1b_candidates)
        for i, row in article[abstract_flag].head(2).iterrows():
            if not pred_b:
                pred_b = check_1b_by_map(row['sentence'], abs_stru)
                if pred_b:
                    article_prediction.append([pmcid, row['sentence_id'], '1b'])
            else:
                break
    article_prediction = pd.DataFrame(article_prediction, columns=['PMCID', 'sentence_id', '1a_1b'])
    article_prediction = article_prediction.groupby(['PMCID', 'sentence_id']).agg(lambda x: x.tolist()).reset_index()

    # bert predictions
    batch_num = len(sentences) // 4 + (len(sentences) % 4)
    all_result = []
    progress = tqdm.tqdm(total=batch_num, ncols=75)

    for batch in DataLoader(sentences, batch_size=4, shuffle=False,
                            collate_fn=colloate_fn):
        result = ort_sess.run(None,
                               {
                                   'text_ids': np.array(batch.text_ids, dtype='int32'),
                                   'attention_mask_text': np.array(batch.attention_mask_text, dtype='float32')
                               })
        result = np.where(result[0] > 0.5, 1, 0)
        all_result.extend(result.tolist())
        progress.update(1)
    progress.close()

    labels = []
    for inst in all_result:
        label = []
        for j in range(len(inst)):
            if inst[j] == 1:
                if column[j] not in ['3b', '6b']:
                    label.append(column[j])
        if len(label) == 0:
            label.append('0')
        labels.append(label)
    file['label'] = labels
    file = file.merge(rule_based_prediction, on=['PMCID', 'sentence_id'], how='outer')
    file = file.merge(article_prediction, on=['PMCID', 'sentence_id'], how='outer')
    file['label'] = file.apply(lambda x: merge(x['label'], x['3b_6b'], x['1a_1b']), axis=1)
    file = file.drop(columns=['3b_6b', '1a_1b'])
    file.to_csv(saved_path, index=False)
    print("INFO: FILE SAVED")

def main_spirit(model_path, tokenizer_name, csv_path, saved_path, column, mapping):
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    ort_sess = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])

    sentences, file = contextual_load(csv_path, tokenizer_name)
    print("INFO: DATA LOADED")

    # for 1a, 1b, 3b, 6b
    rule_based_prediction = []
    for i, row in file.iterrows():
        pred_instance = []
        if check_content(row['sentence'], phrases_3b):
            pred_instance.append('3b')
        if check_content(row['sentence'], phrases_6b):
            pred_instance.append('6b')
        rule_based_prediction.append([row['PMCID'], row['sentence_id'], pred_instance])
    rule_based_prediction = pd.DataFrame(rule_based_prediction, columns=['PMCID', 'sentence_id', '3b_6b'])

    # for 1a and 1b
    article_prediction = []
    abs_stru = structure()
    for pmcid, article in file.groupby('PMCID'):
        pred_a, pred_b = False, False
        title_flag = article['section'].apply(check_title)
        for i, row in article[title_flag].iterrows():
            if not pred_a:
                pred_a = check_1a(row['sentence'])
                if pred_a:
                    article_prediction.append([pmcid, row['sentence_id'], '1a'])
            else:
                break
        abstract_flag = article["section"].apply(find_1b_candidates)
        for i, row in article[abstract_flag].head(2).iterrows():
            if not pred_b:
                pred_b = check_1b_by_map(row['sentence'], abs_stru)
                if pred_b:
                    article_prediction.append([pmcid, row['sentence_id'], '1b'])
            else:
                break
    article_prediction = pd.DataFrame(article_prediction, columns=['PMCID', 'sentence_id', '1a_1b'])
    article_prediction = article_prediction.groupby(['PMCID', 'sentence_id']).agg(lambda x: x.tolist()).reset_index()

    # bert predictions
    batch_num = len(sentences) // 4 + (len(sentences) % 4)
    all_result = []
    progress = tqdm.tqdm(total=batch_num, ncols=75)

    for batch in DataLoader(sentences, batch_size=4, shuffle=False,
                            collate_fn=colloate_fn):
        result = ort_sess.run(None,
                               {
                                   'text_ids': np.array(batch.text_ids, dtype='int32'),
                                   'attention_mask_text': np.array(batch.attention_mask_text, dtype='float32')
                               })
        result = np.where(result[0] > 0.5, 1, 0)
        all_result.extend(result.tolist())
        progress.update(1)
    progress.close()

    labels = []
    for inst in all_result:
        label = []
        for j in range(len(inst)):
            if inst[j] == 1:
                if column[j] not in ['3b', '6b']:
                    label.append(column[j])
        if len(label) == 0:
            label.append('0')
        labels.append(label)
    file['label'] = labels
    file = file.merge(rule_based_prediction, on=['PMCID', 'sentence_id'], how='outer')
    file = file.merge(article_prediction, on=['PMCID', 'sentence_id'], how='outer')
    file['label'] = file.apply(lambda x: merge(x['label'], x['3b_6b'], x['1a_1b']), axis=1)
    file = file.drop(columns=['3b_6b', '1a_1b'])

    convert_dict = dict()
    for i, row in mapping[['SPIRIT', 'CONSORT']].dropna().iterrows():
        convert_dict[row.tolist()[1]] = row.tolist()[0]

    def convert(row):
        converted = []
        for item in row:
            if item in convert_dict.keys():
                converted.append(convert_dict[item])
        return list(set(converted))

    file["label"] = file['label'].apply(lambda x: convert(x))
    file.to_csv(saved_path, index=False)
    print("INFO: FILE SAVED")

def get_elapsed_time(start_time, end_time):

    runtime = (end_time - start_time).seconds

    hours, remainder = divmod(runtime, 3600)
    minutes, seconds = divmod(remainder, 60)

    time_str = ""

    if hours:
        time_str += f"{hours} hours, "

    if minutes:
        time_str += f"{minutes} minutes, "

    if seconds:
        time_str += f"{seconds} seconds"

    return time_str


if __name__ == '__main__':
    # path to the model
    model_path = 'optimized_model_v2.onnx'
    tokenizer_name = '/Users/jianglan/Documents/server/bionlp/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/'
    # path to the text folder
    folder_path = '/Users/jianglan/RCT-Transparency/inference/all_CONSORT_manual_data.csv'
    # path to the file to be saved
    saved_path = 'prediction_covid.csv'
    start_time = datetime.datetime.now()
    guideline = "consort"
    # choose guideline from "consort" and "spirit"
    column = ['10', '11a', '11b', '12a', '12b', '13a', '13b', '14a', '14b', '15', '16', '17a', '17b',
              '18', '19', '20', '21', '22', '23', '24', '25', '2b', '3a', '3b', '4a', '4b', '5',
              '6a', '6b', '7a', '7b', '8a', '8b', '9']

    if guideline == "consort":
        main_consort(model_path, tokenizer_name, folder_path, saved_path, column)

    elif guideline == "spirit":
        mapping = pd.read_csv('mapping.csv', header=0)
        main_spirit(model_path, tokenizer_name, folder_path, saved_path, column, mapping)


    end_time = datetime.datetime.now()
    print("INFO: Time spent: ", get_elapsed_time(start_time, end_time))
