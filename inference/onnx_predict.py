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
import datetime
import onnxruntime as ort


batch_fields_contextual = ['text_ids', 'attention_mask_text']
Batch_contextual = namedtuple('Batch', field_names=batch_fields_contextual,
                              defaults=[None] * len(batch_fields_contextual))
instance_fields = ['text_ids', 'attention_mask_text']
Instance = namedtuple('Instance', field_names=instance_fields,
                      defaults=[None] * len(instance_fields))


def process_contextual_for_bert(tokenizer, data, max_length=128):
    instances = []
    for i in data:
        text_ids = tokenizer.encode(i[0],
                                    add_special_tokens=False,
                                    truncation=True,
                                    max_length=max_length)
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

        instance = Instance(text_ids=text_ids_all, attention_mask_text=attn_mask)
        instances.append(instance)
    return instances


def contextual_load(path, config):
    file_data = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames[:50]:
            f = open(dirpath + '/' + filename)
            f = f.readline()
            if len(f) != 0:
                f = segment(f)
                for s in f:
                    file_data.append([dirpath.split('/')[-1], filename, s['sentence'], s['span']])
    file = pd.DataFrame(file_data, columns=['folder', 'file', 'sentence', 'span'])
    print("INFO: DATA PARSED")

    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name, do_lower_case=True)
    data = process_contextual_for_bert(tokenizer, file_data)
    return data, file


def collate_fn_contextual(batch, gpu=False):
    batch_text_idxs = []
    batch_attention_masks_text = []
    for inst in batch:
        batch_text_idxs.append(inst.text_ids)
        batch_attention_masks_text.append(inst.attention_mask_text)

    batch_text_idxs = torch.LongTensor(batch_text_idxs)
    batch_attention_masks_text = torch.FloatTensor(batch_attention_masks_text)
    return Batch_contextual(
        text_ids=batch_text_idxs,
        attention_mask_text=batch_attention_masks_text,
    )


def main(model_path, data_path, saved_path, column):
    config = Config()
    data, file = contextual_load(data_path, config)
    print("INFO: DATA LOADED")

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    ort_sess = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
    batch_num = len(data) // config.eval_batch_size + (len(data) % config.eval_batch_size)
    progress = tqdm.tqdm(total=batch_num)
    all_result = []
    with torch.no_grad():
        for batch in DataLoader(data, batch_size=config.eval_batch_size, shuffle=False,
                                collate_fn=collate_fn_contextual):
            combine = ort_sess.run(None,
                                   {
                                       'text_ids': batch.text_ids.to(torch.int32).numpy(),
                                       'attention_mask_text': batch.attention_mask_text.numpy()
                                   })
            combine = np.where(combine[0] > 0.5, 1, 0)
            all_result.extend(combine.tolist()[:config.eval_batch_size])
            progress.update(1)
    progress.close()

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
    model_path = 'optimized_model.onnx'
    # path to the text folder
    data_path = '/efs/lanj3/comparison_set_individual_files/full_texts'
    # path to the file to be saved
    save_path = '/efs/lanj3/prediction_covid.csv'
    start = datetime.datetime.now()
    labels_to_predict = ['2a', '2b', '3a', '3b', '4a', '4b', '5', '6a', '6b', '7a', '7b', '8a', '8b', '9', '10',
                         '11a', '11b', '12a', '12b', '13a', '13b', '14a', '14b', '15', '16', '17a', '17b', '18',
                         '19', '20', '21', '22', '23', '24', '25']
    main(model_path, data_path, save_path, labels_to_predict)
    end = datetime.datetime.now()
    print("INFO: Time spent: ", get_elapsed_time(start, end))
