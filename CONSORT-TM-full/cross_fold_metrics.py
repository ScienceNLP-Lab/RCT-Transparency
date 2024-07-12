import numpy as np
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import pandas as pd
import re
from collections import defaultdict
import ast


def custom_metrics_sort(dict2sort):
    """Sorts dictionary based on custom ordering - Orders numbers in descending order first (grouping P/R/F1) and finishes with averages (macro/micro)"""
    custom_order = {c:i for i,c in enumerate(['precision', 'recall', 'f1', 'auc', 'accuracy', '1+ precision', '1+ recall', '1+ f1', 'support'])}
    myKeys = list(dict2sort.keys())
    convert = lambda text: int(text) if text.isdigit() else text
    myKeys = sorted(myKeys, key=lambda x: ([convert(c) for c in re.split('([0-9]+)', x.split(' ', 1)[0])], custom_order.get(x.split(' ', 1)[-1], len(custom_order))))
    sorted_dict = {i: dict2sort[i] for i in myKeys}
    return sorted_dict


def remove_empty(predictions, labels, label_list):
    """Removes column not associated with any label from predictions"""
    empty_idx = []
    for i in range(len(label_list)):
        if not label_list[i]:
            empty_idx.append(i)
    for idx in empty_idx:
        [j.pop(idx) for j in predictions]
        [j.pop(idx) for j in labels]
        label_list.pop(idx)
    return predictions, labels, label_list


def calculate_multilabel_instance_metrics(predictions, labels, label_list):
    """
    Calculate a variety of performance metrics (precision, recall, f1, auc) for multi-label data.
    :param predictions: list of lists or similar - binary predictions from model
    :param labels: list of lists or similar - true labels
    :param label_list: list of strings with name of each label
    :return: dictionary of metrics
    """
    predictions, labels, label_list = remove_empty(predictions, labels, label_list)
    # Averaged metrics - micro and macro
    precision_micro_average = precision_score(y_true=labels, y_pred=predictions, average='micro', zero_division=0)
    precision_macro_average = precision_score(y_true=labels, y_pred=predictions, average='macro', zero_division=0)
    recall_micro_average = recall_score(y_true=labels, y_pred=predictions, average='micro', zero_division=0)
    recall_macro_average = recall_score(y_true=labels, y_pred=predictions, average='macro', zero_division=0)
    f1_micro_average = f1_score(y_true=labels, y_pred=predictions, average='micro', zero_division=0)
    f1_macro_average = f1_score(y_true=labels, y_pred=predictions, average='macro', zero_division=0)

    metrics = {'micro f1': f1_micro_average,
                'macro f1': f1_macro_average,
                'micro precision': precision_micro_average,
                'macro precision': precision_macro_average,
                'micro recall': recall_micro_average,
                'macro recall': recall_macro_average,
                }
    # Individual label metrics
    clf_dict = classification_report(labels, predictions, target_names=label_list, digits=4, zero_division=0, output_dict=True)
    for i, lab in enumerate(label_list):
        if not lab.isspace() and lab:
            # P, R, F1
            metrics[f'{lab} precision'] = clf_dict[lab]['precision']
            metrics[f'{lab} recall'] = clf_dict[lab]['recall']
            metrics[f'{lab} f1'] = clf_dict[lab]['f1-score']

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
        df[label] = np.where(df[label] > 0, df.index + 1, 0)  # no index should be 0 since that is used for no predictions; start at 1
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
#def calc_set_similarity(agg_set_labels_lol, agg_set_preds_lol, label_list):
def calc_set_similarity(agg_set_preds_lol, agg_set_labels_lol, label_list):
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
    perf_dict[f'macro 1+ precision'] = sum(macro_p_plus_1)/len(macro_p_plus_1)
    perf_dict[f'macro 1+ recall'] = sum(macro_r_plus_1)/len(macro_r_plus_1)
    perf_dict[f'macro 1+ f1'] = sum(macro_f1_plus_1)/len(macro_f1_plus_1)

    # add micro averages
    micro_p_plus_1 = micro_tp_plus_1 / (micro_tp_plus_1 + micro_fp_plus_1)  # precision
    perf_dict[f'micro 1+ precision'] = micro_p_plus_1
    micro_r_plus_1 = micro_tp_plus_1 / (micro_tp_plus_1 + micro_fn_plus_1)  # recall
    perf_dict[f'micro 1+ recall'] = micro_r_plus_1
    micro_f1_plus_1 = (2 * micro_p_plus_1 * micro_r_plus_1) / (micro_p_plus_1 + micro_r_plus_1)
    perf_dict[f'micro 1+ f1'] = micro_f1_plus_1

    sorted_metrics = custom_metrics_sort(perf_dict)

    return sorted_metrics
    

def calculate_multilabel_article_metrics(predictions, labels, label_list, articles):
    """
    Calculate a variety of performance metrics (precision, recall, f1, auc) for multi-label data at the article bag-level.
    :param predictions: list of lists or similar - binary predictions from model
    :param labels: list of lists or similar - true labels
    :param label_list: list of strings with name of each label
    :return: dictionary of metrics
    """
    predictions, labels, label_list = remove_empty(predictions, labels, label_list)
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
    
    metrics = calculate_multilabel_instance_metrics(preds_lol, labels_lol, label_list)

    #set_metrics = calc_set_similarity(agg_set_labels_lol, agg_set_preds_lol, label_list)
    set_metrics = calc_set_similarity(agg_set_preds_lol, agg_set_labels_lol, label_list)
    all_metrics = metrics | set_metrics

    sorted_metrics = custom_metrics_sort(all_metrics)

    return sorted_metrics


def evaluate(valid_result, target_result, list_name, article_ids):
    """Evaluate classification model."""

    instance_report = calculate_multilabel_instance_metrics(valid_result, target_result, list_name)
    article_report = calculate_multilabel_article_metrics(valid_result, target_result, list_name, article_ids)

    return instance_report, article_report


def translate2binary(row, list_labels):
    binary = []
    for lab in list_labels:
        if lab in row:  # checks for each match - not any substring in list
            binary.append(1)
        else:
            binary.append(0)
    return binary


def RemoveIfNoCriteriaPresentInSet(predictions, labels, label_list):
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

    # Remove columns with no labels and no predictions as these will break metrics b/c there is no true positive or false positive
    missing_cols = list(labels_missing_cols.intersection(preds_missing_cols))
    label_list = [lab for lab in label_list if lab not in missing_cols]

    preds_df = preds_df.drop(columns=missing_cols)
    preds_df['prediction'] = preds_df[label_list].values.tolist()
    preds_lol = preds_df['prediction'].to_list()

    labels_df = labels_df.drop(columns=missing_cols)
    labels_df['labels'] = labels_df[label_list].values.tolist()
    labels_lol = labels_df['labels'].to_list()

    return preds_lol, labels_lol, label_list


def process_data(fpath, list_name):
    file = pd.read_csv(fpath)
    file.drop(columns=['Unnamed: 0'], inplace=True)
    file['sid'] = file['sid'].apply(lambda x: x.split('_')[0])
    file['target_result'] = file['target_result'].apply(lambda x: ["_"+str(s) for s in ast.literal_eval(x)])
    ## target_result = true labels
    file['valid_result'] = file['valid_result'].apply(lambda x: ["_"+str(s) for s in ast.literal_eval(x)])
    ## valid_result = predictions

    file['target_result'] = file['target_result'].apply(translate2binary, args=(list_name,))
    file['valid_result'] = file['valid_result'].apply(translate2binary, args=(list_name,))

    return file


if __name__ == '__main__':
    list_name  = ['10', '11a', '11b', '12a', '12b', '13a', '13b', '14a', '14b', '15', '16', '17a', '17b', '18', '19', '20', '21', '22', '23', '24', '25', '2b', '3a', '3b', '4a', '4b', '5', '6a', '6b', '7a', '7b', '8a', '8b', '9']
    list_name = ['_' + s for s in list_name]

    fpaths  = ['preds/fold_1/test_predictions.csv', 'preds/fold_2/test_predictions.csv', 'preds/fold_3/test_predictions.csv', 'preds/fold_4/test_predictions.csv', 'preds/fold_5/test_predictions.csv']
    instance_metric_dicts = defaultdict(list)
    article_metric_dicts = defaultdict(list)
    for fpath in fpaths:  # iterate through model folds [1, 2, 3, 4, 5]
        ## load correct test data based on fold
        data = process_data(fpath, list_name)  # valid_result=predictions; target_result=true labels

        ## remove criteria not present or predicted within label
        preds_lol, labels_lol, list_name = RemoveIfNoCriteriaPresentInSet(data['valid_result'].to_list(), data['target_result'].to_list(), list_name)
        
        ## evaluate using models on correct splits
        instance_report, article_report = evaluate(preds_lol, labels_lol, list_name, data['sid'].to_list())
        #instance_report, article_report = evaluate(data['valid_result'].to_list(), data['target_result'].to_list(), list_name, data['sid'].to_list())

        # add performances to create averages across folds
        for key in instance_report:
            instance_metric_dicts[key].append(instance_report[key])
        for key in article_report:
            article_metric_dicts[key].append(article_report[key])

    # Pretty print metrics (avg, std)
    art_keys = list(article_metric_dicts.keys())
    print(f"{'Label':20s}{'Instance':20s}\t{'Article':20s}")
    blank = 'N/A'
    for art_keys in article_metric_dicts:
        try:
            print(f"{art_keys:20s}{sum(instance_metric_dicts[art_keys])/len(instance_metric_dicts[art_keys]):9f}, {np.std(instance_metric_dicts[art_keys]):9f}\t{sum(article_metric_dicts[art_keys])/len(article_metric_dicts[art_keys]):9f}, {np.std(article_metric_dicts[art_keys]):9f}")
        except:
            print(f"{art_keys:20s}{blank:20s}\t{sum(article_metric_dicts[art_keys])/len(article_metric_dicts[art_keys]):9f}, {np.std(article_metric_dicts[art_keys]):9f}")
        print('-'*60)
