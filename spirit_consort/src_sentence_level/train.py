import numpy as np
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score, \
    zero_one_loss, accuracy_score
import pandas as pd
from transformers import AdamW, get_linear_schedule_with_warmup
from models.bert_model import BERT
from config import Config
from data import collate_fn, contextual_load, single_load
import torch
from torch.utils.data import DataLoader
import tqdm
import re
import time, os, json
import random
import pickle
from collections import defaultdict


def custom_metrics_sort(dict2sort):
    """Sorts dictionary based on custom ordering - Orders numbers in descending order first (grouping P/R/F1) and finishes with averages (macro/micro)"""
    custom_order = {c: i for i, c in enumerate(
        ['precision', 'recall', 'f1', 'auc', 'accuracy', '1+ precision', '1+ recall', '1+ f1', 'support'])}
    myKeys = list(dict2sort.keys())
    convert = lambda text: int(text) if text.isdigit() else text
    myKeys = sorted(myKeys, key=lambda x: ([convert(c) for c in re.split('([0-9]+)', x.split(' ', 1)[0])],
                                           custom_order.get(x.split(' ', 1)[-1], len(custom_order))))
    sorted_dict = {i: dict2sort[i] for i in myKeys}
    return sorted_dict


def remove_empty(predictions_logits, predictions, labels, label_list):
    """Removes column not associated with any label from predictions"""
    empty_idx = []
    for i in range(len(label_list)):
        if not label_list[i]:
            empty_idx.append(i)
    for idx in empty_idx:
        [j.pop(idx) for j in predictions_logits]
        [j.pop(idx) for j in predictions]
        [j.pop(idx) for j in labels]
        label_list.pop(idx)
    return predictions_logits, predictions, labels, label_list


def remove_if_no_criteria_present(predictions_logits, predictions, labels, label_list):
    """Removes criteria without any true positive instances or predictions within the inputted test set."""
    # get crieria names w/ zeros to remove from predictions and list of labels (label_list)
    preds_dict = {'prediction': predictions}
    preds_df = pd.DataFrame(preds_dict)
    preds_df[label_list] = pd.DataFrame(preds_df['prediction'].tolist(), index=preds_df.index)
    preds_df = preds_df.drop('prediction', axis=1)
    preds_missing_cols = set(preds_df.columns[(preds_df == 0).all()])

    # Detect if any labels have no positive instances in test set - we remove these criteria
    labels_dict = {'label': labels}
    labels_df = pd.DataFrame(labels_dict)
    labels_df[label_list] = pd.DataFrame(labels_df['label'].tolist(), index=labels_df.index)
    labels_df = labels_df.drop('label', axis=1)
    labels_missing_cols = set(labels_df.columns[(labels_df == 0).all()])

    # Format prediction logits
    preds_logits_dict = {'prediction': predictions_logits}
    preds_logits_df = pd.DataFrame(preds_logits_dict)
    preds_logits_df[label_list] = pd.DataFrame(preds_logits_df['prediction'].tolist(), index=preds_logits_df.index)
    preds_logits_df = preds_logits_df.drop('prediction', axis=1)

    # Remove columns with no labels and no predictions as these will break metrics b/c there is no true positive or false positive
    missing_cols = list(labels_missing_cols.intersection(preds_missing_cols))
    label_list = [lab for lab in label_list if lab not in missing_cols]

    preds_df = preds_df.drop(columns=missing_cols)
    preds_df['prediction'] = preds_df[label_list].values.tolist()
    preds_lol = preds_df['prediction'].to_list()

    labels_df = labels_df.drop(columns=missing_cols)
    labels_df['labels'] = labels_df[label_list].values.tolist()
    labels_lol = labels_df['labels'].to_list()

    preds_logits_df = preds_logits_df.drop(columns=missing_cols)
    preds_logits_df['prediction'] = preds_logits_df[label_list].values.tolist()
    preds_logits_lol = preds_logits_df['prediction'].to_list()

    return preds_logits_lol, preds_lol, labels_lol, label_list


def calculate_multilabel_instance_metrics(predictions_logits, predictions, labels, label_list, sentence=False):
    """
    Calculate a variety of performance metrics (precision, recall, f1, auc) for multi-label data.
    :param predictions_logits: list of lists or similar - prediction logits from model
    :param predictions: list of lists or similar - binary predictions from model
    :param labels: list of lists or similar - true labels
    :param label_list: list of strings with name of each label
    :return: dictionary of metrics
    """
    if sentence:
        predictions_logits, predictions, labels, label_list = remove_empty(predictions_logits, predictions, labels,
                                                                           label_list)
        predictions_logits, predictions, labels, label_list = remove_if_no_criteria_present(predictions_logits,
                                                                                            predictions, labels,
                                                                                            label_list)

    predictions_logits_arr = np.array(predictions_logits)
    labels_arr = np.array([np.array(xi) for xi in labels])
    # Averaged metrics - micro and macro
    precision_micro_average = precision_score(y_true=labels, y_pred=predictions, average='micro', zero_division=0)
    precision_macro_average = precision_score(y_true=labels, y_pred=predictions, average='macro', zero_division=0)
    recall_micro_average = recall_score(y_true=labels, y_pred=predictions, average='micro', zero_division=0)
    recall_macro_average = recall_score(y_true=labels, y_pred=predictions, average='macro', zero_division=0)
    f1_micro_average = f1_score(y_true=labels, y_pred=predictions, average='micro', zero_division=0)
    f1_macro_average = f1_score(y_true=labels, y_pred=predictions, average='macro', zero_division=0)
    zero_one_score_average = 1 - zero_one_loss(labels_arr, predictions)

    metrics = {'micro f1': f1_micro_average,
               'macro f1': f1_macro_average,
               'micro precision': precision_micro_average,
               'macro precision': precision_macro_average,
               'micro recall': recall_micro_average,
               'macro recall': recall_macro_average,
               'zero-one score': zero_one_score_average,
               }
    # Individual label metrics
    clf_dict = classification_report(labels, predictions, target_names=label_list, digits=4, zero_division=0,
                                     output_dict=True)
    for i, lab in enumerate(label_list):
        if not lab.isspace() and lab:
            # P, R, F1
            metrics[f'{lab} precision'] = clf_dict[lab]['precision']
            metrics[f'{lab} recall'] = clf_dict[lab]['recall']
            metrics[f'{lab} f1'] = clf_dict[lab]['f1-score']
            if sentence:
                # AUC for each label
                lab_logits = np.transpose(predictions_logits_arr[:, i])
                lab_labels = np.transpose(labels_arr[:, i])
                try:
                    metrics[f'{lab} auc'] = roc_auc_score(lab_labels, lab_logits, average=None)
                except ValueError:
                    metrics[f'{lab} auc'] = 'Value Error'

    for tar in labels:
        for i, ta in enumerate(tar):
            if ta > 0:
                try:
                    metrics[f'{label_list[i]} support'] += 1
                except KeyError:
                    metrics[f'{label_list[i]} support'] = 1

    sorted_metrics = custom_metrics_sort(metrics)

    return sorted_metrics


# Functions used for set similarity metrics
def get_sent_sets(df, label_list):
    """Replace binary predictions with the predicted sentence index"""
    for label in label_list:
        df[label] = np.where(df[label] > 0, df.index + 1,
                             0)  # no index should be 0 since that is used for no predictions; start at 1
    return df


def to_set(x):
    """Converts all sentence idx series to sets; no predictions returns empty sets"""
    if all(v == 0 for v in x):
        return set()
    else:
        temp = set(x)
        temp.discard(0)
        return temp


# Set similarity calculations - iterate through each label and calculate performance
def calc_set_similarity(agg_set_labels_lol, agg_set_preds_lol, label_list):
    """
    Calculates custom classification metrics using sets
    :param agg_set_labels_lol: list of lists or similar containing sets of labeled sentence indices - true labels
    :param agg_set_preds_lol: list of lists or similar containing sets of predicted sentence indices - predictions from model
    :param label_list: list of strings with name of each label
    :return: sorted dictionary of metrics
    """
    perf_dict = dict()
    macro_p_plus_1, macro_r_plus_1, macro_f1_plus_1 = [], [], []
    micro_tp_plus_1, micro_fp_plus_1, micro_fn_plus_1 = 0, 0, 0
    for i, lab in enumerate(label_list):
        if lab:
            tp_plus_1, fp_plus_1, fn_plus_1 = 0, 0, 0
            for j, sets in enumerate(agg_set_labels_lol):
                tp, fp, fn = 0, 0, 0
                lab_labels = agg_set_labels_lol[j][i]
                lab_preds = agg_set_preds_lol[j][i]
                if bool(lab_labels) or bool(lab_preds):  # check if both empty - no preds/true labels
                    tp = len(lab_labels & lab_preds)  # number of labels - TP is sentence id in both true/preds
                    fp = len(lab_preds - lab_labels)  # FP is sentence id in preds, not true
                    fn = len(lab_labels - lab_preds)  # FN is sentence id is in true, not preds
                    if tp > 0:
                        tp_plus_1 += 1
                        micro_tp_plus_1 += 1
                    elif fp > 0:
                        fp_plus_1 += 1
                        micro_fp_plus_1 += 1
                    elif fn > 0:
                        fn_plus_1 += 1
                        micro_fn_plus_1 += 1

            # 1+ Metrics
            try:
                p_plus_1 = tp_plus_1 / (tp_plus_1 + fp_plus_1)  # precision
            except ZeroDivisionError:
                p_plus_1 = 0.0
            try:
                r_plus_1 = tp_plus_1 / (tp_plus_1 + fn_plus_1)  # recall
            except ZeroDivisionError:
                r_plus_1 = 0.0
            perf_dict[f'{lab} 1+ precision'] = p_plus_1
            perf_dict[f'{lab} 1+ recall'] = r_plus_1
            try:
                f1_plus_1 = (2 * p_plus_1 * r_plus_1) / (p_plus_1 + r_plus_1)
            except ZeroDivisionError:
                f1_plus_1 = 0.0
            perf_dict[f'{lab} 1+ f1'] = f1_plus_1
            macro_p_plus_1.append(p_plus_1)
            macro_r_plus_1.append(r_plus_1)
            macro_f1_plus_1.append(f1_plus_1)

    # add macro averages
    perf_dict[f'macro 1+ precision'] = sum(macro_p_plus_1) / len(macro_p_plus_1)
    perf_dict[f'macro 1+ recall'] = sum(macro_r_plus_1) / len(macro_r_plus_1)
    perf_dict[f'macro 1+ f1'] = sum(macro_f1_plus_1) / len(macro_f1_plus_1)

    # add micro averages
    micro_p_plus_1 = micro_tp_plus_1 / (micro_tp_plus_1 + micro_fp_plus_1)  # precision
    perf_dict[f'micro 1+ precision'] = micro_p_plus_1
    micro_r_plus_1 = micro_tp_plus_1 / (micro_tp_plus_1 + micro_fn_plus_1)  # recall
    perf_dict[f'micro 1+ recall'] = micro_r_plus_1
    micro_f1_plus_1 = (2 * micro_p_plus_1 * micro_r_plus_1) / (micro_p_plus_1 + micro_r_plus_1)
    perf_dict[f'micro 1+ f1'] = micro_f1_plus_1

    sorted_metrics = custom_metrics_sort(perf_dict)

    return sorted_metrics


def calculate_multilabel_section_metrics(predictions_logits, predictions, labels, label_list, sections, articles):
    """
    Calculate a variety of performance metrics (precision, recall, f1, auc) for multi-label data at the section bag-level.
    :param predictions_logits: list of lists or similar - prediction logits from model
    :param predictions: list of lists or similar - binary predictions from model
    :param labels: list of lists or similar - true labels
    :param label_list: list of strings with name of each label
    :return: dictionary of metrics
    """
    predictions_logits, predictions, labels, label_list = remove_empty(predictions_logits, predictions, labels,
                                                                       label_list)
    predictions_logits, predictions, labels, label_list = remove_if_no_criteria_present(predictions_logits, predictions,
                                                                                        labels, label_list)
    # Aggregate and format labels
    labels_dict = {'article': articles, 'section': sections, 'label': labels}
    labels_df = pd.DataFrame(labels_dict)
    labels_df[label_list] = pd.DataFrame(labels_df['label'].tolist(), index=labels_df.index)

    labels_df = labels_df.drop('label', axis=1)
    agg_labels_df = labels_df.groupby(['article', 'section']).max()
    agg_labels_df['labels'] = agg_labels_df[label_list].values.tolist()
    labels_lol = agg_labels_df['labels'].to_list()

    # For each label, make a set of sentence IDs that are positively labeled within that section/article
    labels_df = get_sent_sets(labels_df, label_list)
    agg_set_labels_df = labels_df.groupby(['article', 'section']).agg(to_set)
    agg_set_labels_df['predictions'] = agg_set_labels_df[label_list].values.tolist()
    agg_set_labels_lol = agg_set_labels_df['predictions'].to_list()

    # Aggregate and format predictions
    preds_dict = {'article': articles, 'section': sections, 'predictions': predictions}
    preds_df = pd.DataFrame(preds_dict)
    preds_df[label_list] = pd.DataFrame(preds_df['predictions'].tolist(), index=preds_df.index)
    preds_df = preds_df.drop('predictions', axis=1)
    agg_preds_df = preds_df.groupby(['article', 'section']).max()
    agg_preds_df['predictions'] = agg_preds_df[label_list].values.tolist()
    preds_lol = agg_preds_df['predictions'].to_list()
    # For each label, make a set of sentence IDs that are positively predicted within that section/article
    preds_df = get_sent_sets(preds_df, label_list)
    agg_set_preds_df = preds_df.groupby(['article', 'section']).agg(to_set)
    agg_set_preds_df['predictions'] = agg_set_preds_df[label_list].values.tolist()
    agg_set_preds_lol = agg_set_preds_df['predictions'].to_list()

    # Aggregate and format logits
    logits_dict = {'article': articles, 'section': sections, 'logits': predictions_logits}
    logits_df = pd.DataFrame(logits_dict)
    logits_df[label_list] = pd.DataFrame(logits_df['logits'].tolist(), index=logits_df.index)
    logits_df = logits_df.drop('logits', axis=1)
    agg_logits_df = logits_df.groupby(['article', 'section']).max()
    agg_logits_df['logits'] = agg_logits_df[label_list].values.tolist()
    logits_lol = agg_logits_df['logits'].to_list()

    metrics = calculate_multilabel_instance_metrics(logits_lol, preds_lol, labels_lol, label_list)

    set_metrics = calc_set_similarity(agg_set_labels_lol, agg_set_preds_lol, label_list)

    all_metrics = metrics | set_metrics

    sorted_metrics = custom_metrics_sort(all_metrics)

    return sorted_metrics


def calculate_multilabel_article_metrics(predictions_logits, predictions, labels, label_list, articles):
    """
    Calculate a variety of performance metrics (precision, recall, f1, auc) for multi-label data at the article bag-level.
    :param predictions_logits: list of lists or similar - prediction logits from model
    :param predictions: list of lists or similar - binary predictions from model
    :param labels: list of lists or similar - true labels
    :param label_list: list of strings with name of each label
    :return: dictionary of metrics
    """
    predictions_logits, predictions, labels, label_list = remove_empty(predictions_logits, predictions, labels,
                                                                       label_list)
    predictions_logits, predictions, labels, label_list = remove_if_no_criteria_present(predictions_logits, predictions,
                                                                                        labels, label_list)
    # Aggregate and format labels
    labels_dict = {'article': articles, 'label': labels}
    labels_df = pd.DataFrame(labels_dict)
    labels_df[label_list] = pd.DataFrame(labels_df['label'].tolist(), index=labels_df.index)
    labels_df = labels_df.drop('label', axis=1)
    agg_labels_df = labels_df.groupby(['article']).max()
    agg_labels_df['labels'] = agg_labels_df[label_list].values.tolist()
    labels_lol = agg_labels_df['labels'].to_list()
    # For each label, make a set of sentence IDs that are positively labeled within that section/article
    labels_df = get_sent_sets(labels_df, label_list)
    agg_set_labels_df = labels_df.groupby(['article']).agg(to_set)
    agg_set_labels_df['predictions'] = agg_set_labels_df[label_list].values.tolist()
    agg_set_labels_lol = agg_set_labels_df['predictions'].to_list()

    # Aggregate and format predictions
    preds_dict = {'article': articles, 'predictions': predictions}
    preds_df = pd.DataFrame(preds_dict)
    preds_df[label_list] = pd.DataFrame(preds_df['predictions'].tolist(), index=preds_df.index)
    preds_df = preds_df.drop('predictions', axis=1)
    agg_preds_df = preds_df.groupby(['article']).max()
    agg_preds_df['predictions'] = agg_preds_df[label_list].values.tolist()
    preds_lol = agg_preds_df['predictions'].to_list()
    # For each label, make a set of sentence IDs that are positively predicted within that section/article
    preds_df = get_sent_sets(preds_df, label_list)
    agg_set_preds_df = preds_df.groupby(['article']).agg(to_set)
    agg_set_preds_df['predictions'] = agg_set_preds_df[label_list].values.tolist()
    agg_set_preds_lol = agg_set_preds_df['predictions'].to_list()

    # Aggregate and format logits
    logits_dict = {'article': articles, 'logits': predictions_logits}
    logits_df = pd.DataFrame(logits_dict)
    logits_df[label_list] = pd.DataFrame(logits_df['logits'].tolist(), index=logits_df.index)
    logits_df = logits_df.drop('logits', axis=1)
    agg_logits_df = logits_df.groupby(['article']).max()
    agg_logits_df['logits'] = agg_logits_df[label_list].values.tolist()
    logits_lol = agg_logits_df['logits'].to_list()

    metrics = calculate_multilabel_instance_metrics(logits_lol, preds_lol, labels_lol, label_list)

    set_metrics = calc_set_similarity(agg_set_labels_lol, agg_set_preds_lol, label_list)
    all_metrics = metrics | set_metrics

    sorted_metrics = custom_metrics_sort(all_metrics)

    return sorted_metrics


def evaluate(model, loss_fn, data, config, batch_num, epoch, name, list_name, log_dir=None):
    """Make predictions and evaluate classification model."""
    progress = tqdm.tqdm(total=batch_num, ncols=75, desc='{} {}'.format(name, epoch))
    logit_result = []
    valid_result = []
    target_result = []
    article_ids = []
    sections = []
    running_loss = 0.0
    for batch in DataLoader(data, batch_size=config.eval_batch_size, shuffle=False, collate_fn=collate_fn):
        model.eval()
        target = batch.labels
        logits = model(batch)
        loss = loss_fn(logits, target)

        running_loss += loss.item()
        if config.use_gpu:
            target = target.cpu().data
            logits = logits.cpu().data

        binary_preds = np.where(logits.numpy() > 0.5, 1, 0)

        logit_result.extend(logits.tolist())
        valid_result.extend(binary_preds.tolist())
        target_result.extend(target.tolist())

        sections.extend(batch.section)
        article_ids.extend(batch.PMCID)

        progress.update(1)
    progress.close()

    with open('temp_preds.pkl', 'wb') as file:
        temp_dict = {'logits': logit_result, 'valid_result': valid_result, 'target_result': target_result,
                     'list_name': list_name, 'sections': sections, 'article_ids': article_ids}
        pickle.dump(temp_dict, file)

    instance_report = calculate_multilabel_instance_metrics(logit_result, valid_result, target_result, list_name,
                                                            sentence=True)
    section_report = calculate_multilabel_section_metrics(logit_result, valid_result, target_result, list_name,
                                                          sections, article_ids)
    article_report = calculate_multilabel_article_metrics(logit_result, valid_result, target_result, list_name,
                                                          article_ids)

    # Pretty print metrics
    sec_keys = list(section_report.keys())
    art_keys = list(article_report.keys())
    print(f"{'Label' : <70}{'Instance' : ^20}{'Section' : ^20}{'Article' : >20}")
    blank = 'N/A'
    for i, inst_keys in enumerate(list(section_report.keys())):
        try:
            print(
                f"{inst_keys : <70}{instance_report[inst_keys]: ^20}{section_report[sec_keys[i]]: ^20}{article_report[art_keys[i]]: >20}")
        except:
            print(f"{inst_keys : <70}{blank: ^20}{section_report[sec_keys[i]]: ^20}{article_report[art_keys[i]]: >20}")

    return (instance_report, section_report, article_report), running_loss / len(data), (
    valid_result, target_result, sections, article_ids)


def train_bert(config, save):
    """Train BERT-based sentence classification model."""
    train, valid, test, labels = contextual_load(config)

    if save:
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_home_dir = os.path.join(config.log_path, timestamp)
        os.mkdir(log_home_dir)
        log_file = os.path.join(log_home_dir, 'log.txt')
        best_model = os.path.join(log_home_dir, 'best.mdl')
        with open(log_file, 'w', encoding='utf-8') as w:
            w.write(json.dumps(config.to_dict()) + '\n')

    use_gpu = config.use_gpu
    if use_gpu and config.gpu_device >= 0:
        torch.cuda.set_device(config.gpu_device)

    model = BERT(config, len(labels))
    model.load_bert(config.bert_model_name)
    batch_num = len(train) // config.batch_size
    total_steps = batch_num * config.max_epoch
    test_batch_num = len(test) // config.eval_batch_size + (len(test) % config.eval_batch_size != 0)
    if use_gpu:
        model.cuda()
    param_groups = [
        {
            'params': [p for n, p in model.named_parameters() if n.startswith('bert')],
            'lr': config.bert_learning_rate, 'weight_decay': config.bert_weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if not n.startswith('bert')],
            'lr': config.learning_rate, 'weight_decay': config.weight_decay
        },
    ]

    optimizer = AdamW(params=param_groups, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    loss_fn = torch.nn.BCELoss()
    best_f1, best_epoch = 0, 0

    for epoch in range(config.max_epoch):
        running_loss = 0.0
        progress = tqdm.tqdm(total=batch_num, ncols=75, desc='Train {}'.format(epoch))
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(DataLoader(
                train, batch_size=config.batch_size,
                shuffle=True, collate_fn=collate_fn)):
            optimizer.zero_grad()
            model.train()
            prediction = model(batch)
            loss = loss_fn(prediction, batch.labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            progress.update(1)

        progress.close()
        print('INFO: Training Loss is ', round(running_loss / len(train), 4))
        train_loss = running_loss / len(train)

        report_valid, valid_loss, pred_targets = evaluate(model, loss_fn, valid, config, test_batch_num, epoch, 'VALID',
                                                          labels, log_home_dir)
        valid_f1 = report_valid[0]['micro f1']  # instance level micro f1

        if valid_f1 > best_f1:
            best_f1 = valid_f1
            best_epoch = epoch

            report_test, test_loss, pred_targets = evaluate(model, loss_fn, test, config, test_batch_num, epoch, 'TEST',
                                                            labels, log_home_dir)
            if save:
                result = json.dumps({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                    'test_loss': test_loss,
                    'instance_report_valid': report_valid[0],
                    'section_report_valid': report_valid[1],
                    'article_report_valid': report_valid[2],
                    'instance_report_test': report_test[0],
                    'section_report_test': report_test[1],
                    'article_report_test': report_test[2],
                })
                torch.save(dict(model=model.state_dict(), config=config.to_dict()), best_model)
                with open(log_file, 'a', encoding='utf-8') as w:
                    w.write(result + '\n')
                print('INFO: Log file: ', log_file)
        else:
            if save:
                result = json.dumps({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                    'instance_report_valid': report_valid[0],
                    'section_report_valid': report_valid[1],
                    'article_report_valid': report_valid[2]
                })
                with open(log_file, 'a', encoding='utf-8') as w:
                    w.write(result + '\n')
                print('INFO: Log file: ', log_file)
    if save:
        best = json.dumps({'best epoch': best_epoch})
        with open(log_file, 'a', encoding='utf-8') as w:
            w.write(best + '\n')


if __name__ == '__main__':
    save = 1
    config = Config.from_json_file('models/config.json')

    # set random seed
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    train_bert(config, save)
