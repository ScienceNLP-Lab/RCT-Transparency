import torch
from collections import namedtuple
from transformers import BertTokenizer, BertModel
from collections import Counter
import ast, math, random
# import consort.data_aug.UMLS_EDA.augment4class as augment4class
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import os, sys

batch_fields = ['text_ids', 'attention_mask_text', 'labels', 'index', 'sent_id', 'rltv_id',
                'section_ids', 'attention_mask_section', 'sec_token_len', "PMCID"]
batch_fields_contextual = ['text_ids', 'attention_mask_text', 'labels', 'index', 'loss_mask', "PMCID"]

Batch = namedtuple('Batch', field_names=batch_fields,
                   defaults=[None] * len(batch_fields))

Batch_contextual = namedtuple('Batch', field_names=batch_fields_contextual,
                   defaults=[None] * len(batch_fields_contextual))

instance_fields = ['text', 'text_ids', 'attention_mask_text',
                   'labels', 't_labels', 'PMCID', 'index', 'loss_mask', 'sent_id', 'rltv_id',
                   'section_ids', 'attention_mask_section', 'sec_token_len']

instance_fields_contextual = ['text', 'text_ids', 'attention_mask_text',
                   'labels', 't_labels', 'PMCID', 'index', 'loss_mask']

Instance = namedtuple('Instance', field_names=instance_fields,
                      defaults=[None] * len(instance_fields))

Instance_contextual = namedtuple('Instance', field_names=instance_fields_contextual,
                      defaults=[None] * len(instance_fields_contextual))


def adjust_id(test, max_num):
    for i in range(len(test)):
        if test[i].sent_id > max_num:
            test[i] = test[i]._replace(sent_id=max_num)
    return test

def process_contextual_for_bert(tokenizer, data, max_length=128):
    instances = []

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
        loss_mask = [0] * len([2] + text_ids_before + [3] + text_ids) + [1] + [0] * len(text_ids_after + [3]) + [0] * pad_num


        labels = list(i[1])
        instance = Instance(
            text_ids=text_ids_all,
            attention_mask_text=attn_mask,
            loss_mask=loss_mask,
            labels=labels,
            PMCID=i[-1]
        )

        instances.append(instance)
    return instances

def process_data_for_bert(tokenizer, data, max_length=128):
    instances = []
    c = 0
    for i in data:
        text_ids = tokenizer.encode(i[0],
                                    add_special_tokens=True,
                                    truncation=True,
                                    max_length=max_length)

        pad_num = max_length - len(text_ids)
        attn_mask = [1] * len(text_ids) + [0] * pad_num
        text_ids = text_ids + [0] * pad_num
        labels = list(i[1])
        
        # for adding section headers as separate embedding
        token_sec_lens = []
        section_ids = [2]
        for section in i[6]:
            section_id = tokenizer.encode(section,
                                        add_special_tokens=False,
                                        truncation=True,
                                        max_length=max_length)
            section_ids.extend(section_id)
            token_sec_lens.append(len(section_id))
        section_ids.append(3)
        section_ids = section_ids + [0] * (max_length - len(section_ids))
        attn_mask_section = [1] * len(section_ids) + [0] * (max_length - len(section_ids))
        
        instance = Instance(
            text_ids=text_ids,
            sent_id=i[4], rltv_id=i[5],
            attention_mask_text=attn_mask,
            section_ids=section_ids,
            attention_mask_section=attn_mask_section,
            sec_token_len=token_sec_lens,
            labels=labels,
            t_labels=i[2],
            PMCID=i[3],
            index=c
        )
        instances.append(instance)
        c += 1
    return instances


def colloate_fn(batch, gpu=True):
    batch_text_idxs = []
    batch_attention_masks_text = []
    batch_labels = []
    batch_index = []
    batch_sent_id, batch_rltv_id  = [], []
    batch_section_idxs, batch_attention_masks_section, batch_token_lens = [], [], []
    batch_sid = []
    for inst in batch:
        batch_text_idxs.append(inst.text_ids)
        batch_attention_masks_text.append(inst.attention_mask_text)
        batch_labels.append(inst.labels)
        batch_index.append(inst.index)
        batch_sent_id.append(inst.sent_id)
        batch_rltv_id.append(inst.rltv_id)
        batch_section_idxs.append(inst.section_ids)
        batch_attention_masks_section.append(inst.attention_mask_section)
        batch_token_lens.append(inst.sec_token_len)
        batch_sid.append(inst.PMCID)
    if gpu:
        batch_text_idxs = torch.cuda.LongTensor(batch_text_idxs)
        batch_attention_masks_text = torch.cuda.FloatTensor(batch_attention_masks_text)
        batch_section_idxs = torch.cuda.LongTensor(batch_section_idxs)
        batch_attention_masks_section = torch.cuda.FloatTensor(batch_attention_masks_section)
        batch_labels = torch.cuda.FloatTensor(batch_labels)
        batch_index = torch.cuda.LongTensor(batch_index)
        batch_sent_id = torch.cuda.IntTensor(batch_sent_id)
        batch_rltv_id = torch.cuda.IntTensor(batch_rltv_id)
        batch_sid = batch_sid
    else:
        batch_text_idxs = torch.LongTensor(batch_text_idxs)
        batch_attention_masks_text = torch.FloatTensor(batch_attention_masks_text)
        batch_section_idxs = torch.LongTensor(batch_section_idxs)
        batch_attention_masks_section = torch.FloatTensor(batch_attention_masks_section)
        batch_labels = torch.FloatTensor(batch_labels)
        batch_index = torch.LongTensor(batch_index)
        batch_sent_id = torch.cuda.IntTensor(batch_sent_id)
        batch_rltv_id = torch.cuda.IntTensor(batch_rltv_id)
        batch_sid = batch_sid
    return Batch(
        text_ids=batch_text_idxs,
        attention_mask_text=batch_attention_masks_text,
        section_ids=batch_section_idxs,
        attention_mask_section=batch_attention_masks_section,
        sec_token_len=batch_token_lens,
        sent_id=batch_sent_id,
        rltv_id=batch_rltv_id,
        labels=batch_labels,
        index=batch_index,
        PMCID=batch_sid
    )

def colloate_fn_contextual(batch, gpu=True):
    batch_text_idxs = []
    batch_attention_masks_text = []
    batch_labels = []
    batch_loss_mask = []
    batch_sid = []
    # batch_section_idxs, batch_attention_masks_section = [], []
    for inst in batch:
        batch_text_idxs.append(inst.text_ids)
        batch_attention_masks_text.append(inst.attention_mask_text)
        batch_labels.append(inst.labels)
        batch_loss_mask.append(inst.loss_mask)
        batch_sid.append(inst.PMCID)

    if gpu:
        batch_text_idxs = torch.cuda.LongTensor(batch_text_idxs)
        batch_attention_masks_text = torch.cuda.FloatTensor(batch_attention_masks_text)
        batch_labels = torch.cuda.FloatTensor(batch_labels)
        batch_loss_mask = torch.cuda.FloatTensor(batch_loss_mask)
        batch_sid.append(batch_sid)
    else:
        batch_text_idxs = torch.LongTensor(batch_text_idxs)
        batch_attention_masks_text = torch.FloatTensor(batch_attention_masks_text)
        batch_labels = torch.FloatTensor(batch_labels)
        batch_loss_mask = torch.FloatTensor(batch_loss_mask)
        batch_sid = batch_sid
    return Batch_contextual(
        text_ids=batch_text_idxs,
        attention_mask_text=batch_attention_masks_text,
        labels=batch_labels,
        loss_mask=batch_loss_mask,
        PMCID=batch_sid,
    )


def convert(x):
    label = x.split(', ')
    label = [i.strip() for i in label]
    return label


def unique(section_header):
    headers = []
    for i in section_header:
        if i not in headers:
            headers.append(i)
    return headers

def add_list(item):
    return [item]

def exclude_outliers(l1, lin):
    return_labels = []
    for i in l1:
        if i in lin:
            return_labels.append(i)
        else:
            return_labels.append("0")
    return list(set(return_labels))

def contextual_load(path, config, pmids=None, folder_no=None, current_mode=None):
    data = []
    file = pd.read_csv(path, header=0)
    file['CONSORT_item'] = file['CONSORT_Item'].apply(lambda x: ast.literal_eval(x))
    file['section'] = file['section'].apply(lambda x: ast.literal_eval(x))
    file = file[['PMCID', 'sentence_id', 'text', 'CONSORT_item', 'section', 'section.1']]
    # file = file[file["PMCID"].isin(pmids)]
    
    if config.section == "Methods" or config.test_on_specific_section == "Methods":
        file = file[file['section.1'].isin(["Methods", "METHODS", "Patients and methods", "Research Design and Methods", "Subjects and methods", "Materials and methods", "RESEARCH DESIGN AND METHODS"])]
        lin = ["3a", "3b", "4a", "4b", "5", "6a", "6b", "7a", "7b", "8a", "8b", "9", "10", "11a", "11b", "12a", "12b"]

    elif config.section == "Results" or config.test_on_specific_section == "Results":
        file = file[file['section.1'].isin(["Results", "RESULTS"])]
        lin = ["13a", "13b", "14a", "14b", "15", "16", "17a", "17b", "18", "19"]

    elif config.section == "Discussion" or config.test_on_specific_section == "Discussion":
        file = file[file['section.1'].isin(["Discussion", "Study limitations", "DISCUSSION", "CONCLUSIONS", "Conclusions"])]
        lin = ["20", "21", "22"]
    else:
        lin = ['2b', '3a', '3b', "4a", "4b", "5", "6a", "6b", "7a", "7b", "8a", "8b", "9", "10", "11a", "11b", "12a", "12b", "13a", "13b", "14a", "14b", "15", "16", "17a", "17b", "18", "19", \
        "20", "21", "22", "23", "24", "25"]
    
    if current_mode == "train":
        if config.augmentation_mode == 1:
            aug_data = pd.read_csv(config.augmentation_file)
            aug_data["CONSORT_Item"] = aug_data["CONSORTitem"]
        if config.augmentation_mode == 2:
            aug_data = pd.read_csv(config.augmentation_file)
            no_aug_sample_per_folder = int(len(aug_data) / 5)
            print("no_aug_sample_per_folder: ", no_aug_sample_per_folder)
            aug_data = aug_data[no_aug_sample_per_folder * folder_no : no_aug_sample_per_folder * (folder_no + 1)]
            aug_data["CONSORT_Item"] = aug_data["CONSORTitem"]

        if config.augmentation_mode == 3:
            aug_data = pd.read_csv(config.augmentation_file)
            aug_data['SectionHeaders'] = aug_data['SectionHeaders'].apply(lambda x: ast.literal_eval(x))
            aug_data["CONSORT_Item"] = aug_data["CONSORTitem"]

        if config.augmentation_mode == 4:
            aug_data=pd.read_csv("data/eda/" + str(folder_no) + "_augment_all_eda.csv")
            aug_data['SectionHeaders'] = aug_data['section'].apply(lambda x: ast.literal_eval(x))

        if config.augmentation_mode == 5:
            aug_data=pd.read_csv("data/umls-eda/" + str(folder_no) + "_augment_all_eda_umls.csv")
            aug_data['SectionHeaders'] = aug_data['section'].apply(lambda x: ast.literal_eval(x))

        if config.augmentation_mode != 0:
            aug_data['CONSORT_item'] = aug_data['CONSORT_Item'].apply(lambda x: ast.literal_eval(x))

    file['CONSORT_item'] = file.apply(lambda row: exclude_outliers(row['CONSORT_item'], lin), axis=1)
    # create a new column to state max sentence number of each paper
    dict_n_sen = dict()
    file['sentence_id'] = file['sentence_id'].apply(lambda x: int(x[1:]))
    # e.g. sentence_id is "s1" and is changed to 1 by this step
    
    for id, sheet in file.groupby('PMCID'):
        dict_n_sen[id] = max(sheet['sentence_id'])
    file['max_sen'] = file.PMCID.replace(to_replace=dict_n_sen)

    # concat_file = concat_file.drop_duplicates(subset='text')
    label_file = pd.read_csv(path, header=0)
    label_file['CONSORT_item'] = label_file['CONSORT_Item'].apply(lambda x: ast.literal_eval(x))
    file['section'] = file.apply(lambda row: list([row['section.1']] + row.section), axis=1)
    file['section'] = file['section'].map(unique)
    labels = list(set([i for j in label_file['CONSORT_item'] for i in j]))
    labels.remove('1a')
    labels.remove('1b')
    labels.remove('2a')

    labels.sort()
    label = MultiLabelBinarizer()
    list_name = [labels[1:]]
    # list_name = [labels[1:]] is equal to labels.remove('0')
    # the value of list_name equals to [10', '11a', '11b', '12a', '12b', '13a', '13b', '14a', '14b', '15', '16', '17a', '17b', '18', \
    # '19', '20', '21', '22', '23', '24', '25', '2b', '3a', '4a', '4b', '5', '6a', '7a', '7b', '8a', '8b', '9']

    label.fit(list_name)
    train_sentence_ids = []

    for i in range(file.shape[0]):
        if config.header_information_contextual == 2:
            if file.iloc[i].sentence_id == 1:
                before = ''
            else:
                before = ' '.join(file.iloc[i-1].section) + ' ' + file.iloc[i-1].text
            x = ' '.join(file.iloc[i].section) + ' ' + file.iloc[i].text
            y = label.transform([file.iloc[i].CONSORT_item])
            if file.iloc[i].sentence_id == file.iloc[i].max_sen:
                after = ''
            else:
                after = ' '.join(file.iloc[i+1].section) + ' ' + file.iloc[i+1].text
            train_sentence_ids.append(str(file.iloc[i].PMCID) + "_" + str(file.iloc[i].sentence_id))
            data.append([x, y[0], before, after, str(file.iloc[i].PMCID) + "_" + str(file.iloc[i].sentence_id)])
        elif config.header_information_contextual == 0:
            if file.iloc[i].sentence_id == 1:
                before = ''
            else:
                before = file.iloc[i-1].text
            x = file.iloc[i].text
            y = label.transform([file.iloc[i].CONSORT_item])
            if file.iloc[i].sentence_id == file.iloc[i].max_sen:
                after = ''
            else:
                after = file.iloc[i+1].text
            train_sentence_ids.append(str(file.iloc[i].PMCID) + "_" + str(file.iloc[i].sentence_id))
            data.append([x, y[0], before, after, str(file.iloc[i].PMCID) + "_" + str(file.iloc[i].sentence_id)])
        elif config.header_information_contextual == 1:
            if file.iloc[i].sentence_id == 1:
                before = ''
            else:
                before = file.iloc[i-1].text
            x = ' '.join(file.iloc[i].section) + ' ' + file.iloc[i].text
            y = label.transform([file.iloc[i].CONSORT_item])
            if file.iloc[i].sentence_id == file.iloc[i].max_sen:
                after = ''
            else:
                after = file.iloc[i+1].text
            train_sentence_ids.append(str(file.iloc[i].PMCID) + "_" + str(file.iloc[i].sentence_id))
            data.append([x, y[0], before, after, str(file.iloc[i].PMCID) + "_" + str(file.iloc[i].sentence_id)])
    print(data[100])
    print(o)
    if current_mode == "train":
        if config.augmentation_mode in [1, 2]:
            for key, i in aug_data.iterrows():
                x = i["CleanedCurrentSentence"]
                y = label.transform([i["CONSORT_item"]])
                before = i["CleanedPrecedingSentence"]
                after = i["CleanedTrailingSentence"]
                data.append([x, y[0], before, after, "augment_result"])
        if config.augmentation_mode == 3:
            for key, i in aug_data.iterrows():
                x = " ".join(i.SectionHeaders) + " " + i["CleanedRewrittenMiddleSentence"]
                y = label.transform([i["CONSORT_item"]])
                before = i["CleanedRewrittenPrecedingSentence"]
                after = i["CleanedRewrittenTrailingSentence"]
                aug_sentence_id = str(i.PMCID) + "_" + str(i.OriginalSentence).strip("S")
                if aug_sentence_id in train_sentence_ids:
                    data.append([x, y[0], before, after, aug_sentence_id])
        if config.augmentation_mode == 4:
            aug_data_valid = aug_data[aug_data["augmented_set"].notnull()]
            for key, i in aug_data_valid.iterrows():
                x = " ".join(i.section) + " " + i["augmented_set"]
                y = label.transform([i["CONSORT_item"]])
                before = i["augmented_set_preceding"]
                after = i["augmented_set_trailing"]
                aug_sentence_id = i["sentence_id_pmcid"]
                aug_sentence_id = "_".join(aug_sentence_id.split("S"))
                if aug_sentence_id in train_sentence_ids:
                    data.append([x, y[0], before, after, aug_sentence_id])
        if config.augmentation_mode == 5:
            aug_data_valid = aug_data[aug_data["augmented_set"].notnull()]
            for key, i in aug_data_valid.iterrows():
                x = " ".join(i.section) + " " + i["augmented_set"]
                y = label.transform([i["CONSORT_item"]])
                before = i["augmented_set_preceding_umls"]
                after = i["augmented_set_trailing_umls"]
                aug_sentence_id = i["sentence_id_pmcid"]
                aug_sentence_id = "_".join(aug_sentence_id.split("S"))
                if aug_sentence_id in train_sentence_ids:
                    data.append([x, y[0], before, after, aug_sentence_id])

    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name, do_lower_case=True)
    data = process_contextual_for_bert(tokenizer, data)
    
    return data, list_name



def data_load(path, pmids, config, folder_no, current_mode):
    data = []
    file = pd.read_csv(path, header=0)
    file['CONSORT_item'] = file['CONSORT_Item'].apply(lambda x: ast.literal_eval(x))
    file['section'] = file['section'].apply(lambda x: ast.literal_eval(x))
    file = file[file["PMCID"].isin(pmids)]

    file = file[['PMCID', 'sentence_id', 'text', 'CONSORT_item', 'section', 'section.1']]
    print("\n")
    print("original length: ", len(file))
    train_sentence_ids = []
    if config.section == "Methods":
        file = file[file['section.1'].isin(["Methods", "Patients and methods", "Research Design and Methods", "Subjects and methods", "Materials and methods", "RESEARCH DESIGN AND METHODS"])]
        print("current length: ", len(file))
    elif config.section == "Results":
        file = file[file['section.1'].isin(["Results", "RESULTS"])]
        print("current length: ", len(file))
    elif config.section == "Discussion":
        file = file[file['section.1'].isin(["Discussion", "Study limitations", "DISCUSSION", "CONCLUSIONS", "Conclusions"])]
        print("current length: ", len(file))
    # f = lambda x: x.replace('"', '').split(",")

    # create a new column to state max sentence number of each paper
    dict_n_sen = dict()
    file['sentence_id'] = file['sentence_id'].apply(lambda x: int(x[1:]))
    for id, sheet in file.groupby('PMCID'):
        dict_n_sen[id] = max(sheet['sentence_id'])
    file['max_sen'] = file.PMCID.replace(to_replace=dict_n_sen)

    # concat_file = concat_file.drop_duplicates(subset='text')
    label_file = pd.read_csv(path, header=0)
    label_file['CONSORT_item'] = label_file['CONSORT_Item'].apply(lambda x: ast.literal_eval(x))
    file['section'] = file.apply(lambda row: list([row['section.1']] + row.section), axis=1)
    file['section'] = file['section'].map(unique)
    labels = list(set([i for j in label_file['CONSORT_item'] for i in j]))
    labels.remove('1a')
    labels.remove('1b')
    labels.remove('2a')
    # labels.remove('0')

    labels.sort()
    label = MultiLabelBinarizer()
    list_name = [labels[1:]]
    label.fit(list_name)

    for i in file.iterrows():
        if config.section_emb:
            x = i[1].text
        else:
            # format: section header + sentence
            if config.section_header=="both":
                x = ' '.join(i[1].section) + ' ' + i[1].text
            elif config.section_header=="inner":
                x = ' '.join(i[1].section[-1]) + ' ' + i[1].text
            elif config.section_header=="outer":
                x = ' '.join(i[1].section[0]) + ' ' + i[1].text
            elif config.section_header=="none":
                x = i[1].text

        y = label.transform([i[1].CONSORT_item])
        section = i[1].section
        sent_percent_bin = math.floor(i[1].sentence_id/i[1].max_sen*100)/10
        
        # put into 10 bins instead of 11
        if sent_percent_bin == 10:
            sent_percent_bin = 9
        train_sentence_ids.append(str(i[1].PMCID) + "_" + str(i[1].sentence_id))
        data.append([x, y[0], i[1].CONSORT_item, str(i[1].PMCID) + "_" + str(i[1].sentence_id), i[1].sentence_id, sent_percent_bin, section])

    if current_mode == "train":
        print("train mode")
        if config.augmentation_mode == 1:
            aug_data = pd.read_csv(config.augmentation_file)
            aug_data['CONSORT_item'] = aug_data['CONSORTitem'].apply(lambda x: ast.literal_eval(x))

        if config.augmentation_mode == 2:
            aug_data = pd.read_csv(config.augmentation_file)
            no_aug_sample_per_folder = int(len(aug_data) / 5)
            print("no_aug_sample_per_folder: ", no_aug_sample_per_folder)
            aug_data = aug_data[no_aug_sample_per_folder * folder_no : no_aug_sample_per_folder * (folder_no + 1)]
            aug_data['CONSORT_item'] = aug_data['CONSORTitem'].apply(lambda x: ast.literal_eval(x))

        if config.augmentation_mode == 3:
            aug_data = pd.read_csv(config.augmentation_file)
            aug_data['SectionHeaders'] = aug_data['SectionHeaders'].apply(lambda x: ast.literal_eval(x))
            aug_data['CONSORT_item'] = aug_data['CONSORTitem'].apply(lambda x: ast.literal_eval(x))

        if config.augmentation_mode == 4:
            aug_data=pd.read_csv("data/eda/" + str(folder_no) + "_augment_all_eda.csv")
            aug_data['SectionHeaders'] = aug_data['section'].apply(lambda x: ast.literal_eval(x))
            aug_data['CONSORT_item'] = aug_data['CONSORT_Item'].apply(lambda x: ast.literal_eval(x))

        if config.augmentation_mode == 5:
            aug_data=pd.read_csv("data/umls-eda/" + str(folder_no) + "_augment_all_eda_umls.csv")
            aug_data['SectionHeaders'] = aug_data['section'].apply(lambda x: ast.literal_eval(x))
            aug_data['CONSORT_item'] = aug_data['CONSORT_Item'].apply(lambda x: ast.literal_eval(x))


        print("aug_data: ")

        if config.augmentation_mode in [1, 2]:
            for key, i in aug_data.iterrows():
                x = i["CleanedCurrentSentence"]
                y = label.transform([i["CONSORT_item"]])
                data.append([x, y[0], i["CONSORT_item"], 0, 0, 0, [""]])

        elif config.augmentation_mode == 3: 
            for key, i in aug_data.iterrows():
                if config.section_header=="both":
                    x = ' '.join(i.SectionHeaders) + ' ' + i["CleanedRewrittenMiddleSentence"]
                elif config.section_header=="none":
                    x = i["CleanedRewrittenMiddleSentence"]
                y = label.transform([i["CONSORT_item"]])
                aug_sentence_id = str(i.PMCID) + "_" + str(i.OriginalSentence).strip("S")
                if aug_sentence_id in train_sentence_ids:
                    data.append([x, y[0], i["CONSORT_item"], 0, 0, 0, [""]])
        elif config.augmentation_mode in [4, 5]:
            for key, i in aug_data.iterrows():
                x = i["augmented_set"]
                y = label.transform([i["CONSORT_item"]])
                data.append([x, y[0], i["CONSORT_item"], 0, 0, 0, [""]])

    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name, do_lower_case=True)
    data = process_data_for_bert(tokenizer, data)
    return data, list_name

