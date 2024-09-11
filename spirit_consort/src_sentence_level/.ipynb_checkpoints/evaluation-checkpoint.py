import json
import os,ast
import time
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, BertConfig, AdamW, get_linear_schedule_with_warmup
from data import single_load, collate_fn, contextual_load
from train import evaluate
from models.bert_model import BERT
from sklearn.metrics import classification_report, roc_auc_score
from config import Config
from argparse import ArgumentParser

from train import calculate_multilabel_instance_metrics,\
    calculate_multilabel_article_metrics,\
    calculate_multilabel_section_metrics


if __name__ == '__main__':
    path = '/ocean/projects/cis230087p/ljiang2/spirit_consort/20240909_143853/'
    # path = '/ocean/projects/cis230087p/jmenke/spirit_consort-main/20240505_121753/'
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', default='/jet/home/ljiang2/spirit_consort/data/sentences.csv', help='Path to the input file')
    parser.add_argument('-o', '--output', default=path, help='Path to the output file')
    parser.add_argument('-m', '--model', default=path+"best.mdl", help='Path to the model file')
    parser.add_argument('-g', '--guideline', default='SPIRIT', help='SPIRIT or CONSORT')
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--gpu', action='store_true', default=1, help='Use GPU')
    parser.add_argument('-d', '--device', type=int, default=0, help='GPU device index (for multi-GPU machines)')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--save', type=int, default=1, help='whether to save')
    args, unknown = parser.parse_known_args()

    map_location = 'cuda:{}'.format(args.device) if args.gpu else 'cpu'
    if args.gpu:
        torch.cuda.set_device(args.device)

    # set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # load saved weights, config, vocabs and valid patterns
    state = torch.load(args.model, map_location=map_location)

    config = state['config']
    config['data_folder'] = args.input
    if type(config) is dict:
        config = Config.from_dict(config)
    train, valid, test, column = contextual_load(config, 'train')  # train returns all 3 splits

    # initialize model and other parameters
    model = BERT(config, len(column))
    model.load_bert(config.bert_model_name)
    model.load_state_dict(state['model'])

    if args.gpu:
        model.cuda(args.device)

    batch_num = len(test) // args.batch_size + (len(test) % args.batch_size)

    all_result = []
    embedding = []
    targets = []
    progress = tqdm.tqdm(total=batch_num, ncols=75)
    for param in model.parameters():
        param.requires_grad = False

    loss_fn = torch.nn.BCELoss()
    epoch = 0  # does not matter - only used for print statement
    name='test'  # does not matter - only used for print statement

    # eval loop
    print("All items Performance: ")
    reports, avg_running_loss, pred_targets = evaluate(model, loss_fn, test, config, batch_num, epoch, name, column)
    instance_report, section_report, article_report = reports
    result_all = json.dumps({
    'name': 'all items', 
    'avg_running_loss': avg_running_loss, 
    'instance_report': instance_report, 
    'section_report': section_report, 
    'article_report': article_report,
    })
    
    file = pd.DataFrame(zip(pred_targets[3], pred_targets[2], pred_targets[1], pred_targets[0]), columns=['PMCID', 'Section', 'Labels', 'Predictions'])
    
    mapping = pd.read_csv('../analysis/mapping.csv', header=0)
    
    labels = mapping['SPIRIT CONSORT Item'].tolist()
    labels.remove('2_Abstract_structured')
    label_convert = MultiLabelBinarizer()
    label_convert.fit([labels])
    labels = list(label_convert.classes_)
    
    file['all_pred'] = [[labels[j] for j in range(len(i)) if i[j] == 1] for i in file.Predictions]
    file['all_truth'] = [[labels[j] for j in range(len(i)) if i[j] == 1] for i in file.Labels]
    file['all_truth_logit'] = file.Labels
    file['all_pred_logit'] = file.Predictions
    
    convert_dict = dict()
    for i, row in mapping[['SPIRIT CONSORT Item', args.guideline]].dropna().iterrows():
        convert_dict[row.tolist()[0]] = row.tolist()[1]

    def convert(row):
        converted = []
        for item in row:
            if item in convert_dict.keys():
                converted.append(convert_dict[item])
        return list(set(converted))
    
    file[args.guideline+'_pred'] = file['all_truth'].apply(lambda x: convert(x))
    file[args.guideline+'_truth'] = file['all_pred'].apply(lambda x: convert(x))
        
    labels = set([i for i in mapping[args.guideline].dropna()])
    label_convert = MultiLabelBinarizer()
    label_convert.fit([labels])
    labels = list(label_convert.classes_)

    file[args.guideline+'_pred_logit'] = label_convert.transform(file[args.guideline+'_pred']).tolist()
    file[args.guideline+'_truth_logit'] = label_convert.transform(file[args.guideline+'_truth']).tolist()
    

    logit_result = file[args.guideline+'_pred_logit']
    article_ids = file.PMCID.tolist()
    sections = file.Section.tolist()
    target_result = file[args.guideline+'_truth_logit']

    instance_report = calculate_multilabel_instance_metrics(logit_result, logit_result, target_result, labels,
                                                            sentence=True)
    section_report = calculate_multilabel_section_metrics(logit_result, logit_result, target_result, labels, sections,
                                                          article_ids)
    article_report = calculate_multilabel_article_metrics(logit_result, logit_result, target_result, labels, article_ids)

    # Pretty print metrics
    sec_keys = list(section_report.keys())
    art_keys = list(article_report.keys())
    print(args.guideline+" Performance: ")
    print(f"{'Label' : <70}{'Instance' : ^20}{'Section' : ^20}{'Article' : >20}")
    blank = 'N/A'
    for i, inst_keys in enumerate(list(section_report.keys())):
        try:
            print(
                f"{inst_keys : <70}{instance_report[inst_keys]: ^20}{section_report[sec_keys[i]]: ^20}{article_report[art_keys[i]]: >20}")
        except:
            print(f"{inst_keys : <70}{blank: ^20}{section_report[sec_keys[i]]: ^20}{article_report[art_keys[i]]: >20}")
    


    result_guideline = json.dumps({
        'name': args.guideline, 
        'avg_running_loss': avg_running_loss, 
        'instance_report': instance_report, 
        'section_report': section_report, 
        'article_report': article_report,
        })

    if args.save:
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_home_dir = os.path.join(args.output, timestamp)
        os.mkdir(log_home_dir)
        file.to_csv(os.path.join(log_home_dir, 'test.csv'), index=False)
        log_file = os.path.join(log_home_dir, 'log.txt')
        with open(log_file, 'w', encoding='utf-8') as w:
            w.write(json.dumps(config.to_dict()) + '\n')
        with open(log_file, 'a', encoding='utf-8') as w:
            w.write(result_all + '\n')
        with open(log_file, 'a', encoding='utf-8') as w:
            w.write(result_guideline + '\n')
        print('INFO: Log file: ', log_file)




