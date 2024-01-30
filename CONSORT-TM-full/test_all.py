import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
from models.bert_model import BERT
from data import colloate_fn, colloate_fn_contextual, data_load, contextual_load, adjust_id
import torch
from torch.utils.data import DataLoader
import tqdm
import os
from argparse import ArgumentParser
import pickle
from sklearn.metrics import roc_auc_score
import scipy.stats as st

def evaluate(model, Loss, data, config, batch_num, epoch, name, list_name, base_folder_name, folder_no=None, log_dir=None):
    base_folder_name += "_test_on_specific_section=" + str(config.test_on_specific_section)
    progress = tqdm.tqdm(total=batch_num, ncols=75, desc='{} {}'.format(name, epoch))
    valid_result = []
    target_result = []
    sid = []
    sid_auc_logits = []
    running_loss = 0.0
    true_labels = []
    predicted_values = []

    if config.test_on_specific_section == "Methods":
        target_names = ["3a", "3b", "4a", "4b", "5", "6a", "6b", "7a", "7b", "8a", "8b", "9", "10", "11a",
                        "11b", "12a", "12b"]
    elif config.test_on_specific_section == "Results":
        target_names = ["13a", "13b", "14a", "14b", "15", "16", "17a", "17b", "18", "19"]
    elif config.test_on_specific_section == "Discussion":
        target_names = ["20", "21", "22"]
    else:
        target_names = ['2b', '3a', '3b', '4a', '4b', '5', '6a', '6b', '7a', '7b', '8a', '8b', '9', '10',
                        '11a', '11b', '12a', '12b', '13a', '13b', '14a', '14b', '15', '16', '17a', '17b', '18',
                        '19', '20', '21', '22', '23', '24', '25']

    if config.target == "[CLS]":
        if config.mode == "single":
            for batch in DataLoader(data, batch_size=config.eval_batch_size, shuffle=False, collate_fn=colloate_fn):
                model.eval()
                target = batch.labels
                combine = model(batch)
                loss = Loss(combine, target)

                running_loss += loss.item()
                if config.use_gpu:
                    target = target.cpu().data
                    combine = combine.cpu().data

                new_targets = target.tolist()[:4]
                new_predicted_values = combine.tolist()[:4]
                sid_single = batch.PMCID

                for single_index in range(len(new_targets)):
                    if sum(new_targets[single_index]) > 0.1:
                        true_labels.append(new_targets[single_index])
                        predicted_values.append(new_predicted_values[single_index])
                        sid_auc_logits.append(sid_single[single_index])

                combine = np.where(combine.numpy() > 0.5, 1, 0)

                target_result.extend(target.tolist()[:4])
                valid_result.extend(combine.tolist()[:4])

                sid.extend(sid_single[:4])
                progress.update(1)

            true_labels = np.array(true_labels)
            predicted_values = np.array(predicted_values)
            column = list_name[0]

            if config.test_on_specific_section in ["Methods", "Results", "Discussion"]:
                columm_indexes = [i for i, e in enumerate(column) if e in target_names]
                true_labels = true_labels[:, columm_indexes]
                predicted_values = predicted_values[:, columm_indexes]

            print("true_labels.shape: ", true_labels.shape)
            print("predicted_values.shape: ", predicted_values.shape)

            auc_scores_support = pd.DataFrame(list(zip(true_labels, predicted_values)),
                                              columns=["true_labels", "predicted_values"])
            auc_scores_support['folder_on'] = str(folder_no)
            auc_scores_support['sentence_ids'] = sid_auc_logits

            print("base_folder_name: ", base_folder_name)
            auc_scores_support.to_csv(base_folder_name + ".csv", mode='a')
            true_labels = np.array(true_labels)
            predicted_values = np.array(predicted_values)
            micro_auc = roc_auc_score(true_labels, predicted_values, average='micro')
            print("micro_auc: ", micro_auc)
            progress.close()

            report = classification_report(target_result, valid_result, target_names=column, digits=4, zero_division=0,
                                           output_dict=True)
            # column.insert(0, "0")

            targets_final = []
            results_final = []

            # column.insert(0, "0")
            for i in target_result:
                if sum(i) > 0.1:
                    targets_final.append(column[i.index(max(i))])
                else:
                    targets_final.append("0")

            for i in valid_result:
                if sum(i) > 0.1:
                    results_final.append(column[i.index(max(i))])
                else:
                    results_final.append("0")

            return sid, report, running_loss, targets_final, results_final, micro_auc

        elif config.mode == "contextual":
            for batch in DataLoader(data, batch_size=config.eval_batch_size, shuffle=False,
                                    collate_fn=colloate_fn_contextual):
                model.eval()
                target = batch.labels
                combine = model(batch)
                loss = Loss(combine, target)

                running_loss += loss.item()
                if config.use_gpu:
                    target = target.cpu().data
                    combine = combine.cpu().data

                new_targets = target.tolist()[:4]
                new_predicted_values = combine.tolist()[:4]
                sid_single = batch.PMCID

                for single_index in range(len(new_targets)):
                    if sum(new_targets[single_index]) > 0.1:
                        true_labels.append(new_targets[single_index])
                        predicted_values.append(new_predicted_values[single_index])
                        sid_auc_logits.append(sid_single[single_index])

                combine = np.where(combine.numpy() > 0.5, 1, 0)
                target_result.extend(target.tolist()[:4])
                valid_result.extend(combine.tolist()[:4])
                sid.extend(sid_single[:4])
                progress.update(1)
            true_labels = np.array(true_labels)
            predicted_values = np.array(predicted_values)
            column = list_name[0]

            if config.test_on_specific_section in ["Methods", "Results", "Discussion"]:
                columm_indexes = [i for i, e in enumerate(column) if e in target_names]
                print("target_names: ", target_names)
                print("columm_indexes: ", columm_indexes)
                true_labels = true_labels[:, columm_indexes]
                predicted_values = predicted_values[:, columm_indexes]

            print("true_labels.shape: ", true_labels.shape)
            print("predicted_values.shape: ", predicted_values.shape)
            auc_scores_support = pd.DataFrame(list(zip(true_labels, predicted_values)),
                                              columns=["true_labels", "predicted_values"])
            auc_scores_support['folder_on'] = str(folder_no)
            auc_scores_support['sentence_ids'] = sid_auc_logits

            print("base_folder_name: ", base_folder_name)
            auc_scores_support.to_csv(base_folder_name + ".csv", mode='a')
            true_labels = np.array(true_labels)
            predicted_values = np.array(predicted_values)
            micro_auc = roc_auc_score(true_labels, predicted_values, average='micro')

            print("micro_auc: ", micro_auc)
            progress.close()
            column = list_name[0]

            targets_final = []
            results_final = []

            # column.insert(0, "0")
            for i in target_result:
                if sum(i) > 0.1:
                    new_item_targets = [column[j] for j in range(len(i)) if i[j] > 0.5]
                    targets_final.append(new_item_targets)
                else:
                    targets_final.append(["0"])

            for i in valid_result:
                if sum(i) > 0.1:
                    new_item_valid = [column[j] for j in range(len(i)) if i[j] > 0.5]
                    results_final.append(new_item_valid)
                else:
                    results_final.append(["0"])

            # print("targets_final: ", targets_final)
            # print("results_final: ", results_final)

            report = classification_report(target_result, valid_result, target_names=column, digits=4, zero_division=0,
                                           output_dict=True)
            return sid, report, running_loss, targets_final, results_final, micro_auc

    elif config.target == "[SEP]":
        for batch in DataLoader(data, batch_size=config.eval_batch_size, shuffle=False,
                                collate_fn=colloate_fn_contextual):
            model.eval()
            target = batch.labels
            combine = model(batch)
            loss_mask = torch.unsqueeze(batch.loss_mask, 2)
            combine = combine * loss_mask
            combine = torch.sum(combine, 1)
            loss = Loss(combine, target)

            running_loss += loss.item()
            if config.use_gpu:
                target = target.cpu().data
                combine = combine.cpu().data

            new_targets = target.tolist()[:4]
            new_predicted_values = combine.tolist()[:4]

            for single_index in range(len(new_targets)):
                if sum(new_targets[single_index]) > 0.1:
                    true_labels.append(new_targets[single_index])
                    predicted_values.append(new_predicted_values[single_index])

            sid_single = batch.PMCID

            combine = np.where(combine.numpy() > 0.5, 1, 0)
            target_result.extend(target.tolist()[:4])
            valid_result.extend(combine.tolist()[:4])
            sid.extend(sid_single[:4])

            progress.update(1)

        true_labels = np.array(true_labels)
        predicted_values = np.array(predicted_values)
        column = list_name[0]

        if config.test_on_specific_section in ["Methods", "Results", "Discussion"]:
            columm_indexes = [i for i, e in enumerate(column) if e in target_names]
            true_labels = true_labels[:, columm_indexes]
            predicted_values = predicted_values[:, columm_indexes]

        print("true_labels.shape: ", true_labels.shape)
        print("predicted_values.shape: ", predicted_values.shape)
        auc_scores_support = pd.DataFrame(list(zip(true_labels, predicted_values)),
                                          columns=["true_labels", "predicted_values"])
        auc_scores_support['sentence_no'] = str(folder_no)
        auc_scores_support.to_csv(base_folder_name + ".csv", mode='a')

        micro_auc = roc_auc_score(true_labels, predicted_values, average='micro')
        true_labels = np.argmax(true_labels, axis=1)

        print("micro_auc: ", micro_auc)
        progress.close()
        column = list_name[0]
        report = classification_report(target_result, valid_result, target_names=column, digits=4, zero_division=0,
                                       output_dict=True)

        targets_final = []
        results_final = []

        # column.insert(0, "0")
        for i in target_result:
            if sum(i) > 0.1:
                targets_final.append(column[i.index(max(i))])
            else:
                targets_final.append("0")

        for i in valid_result:
            if sum(i) > 0.1:
                results_final.append(column[i.index(max(i))])
            else:
                results_final.append("0")

        return sid, report, running_loss, targets_final, results_final, micro_auc


def test(config):
    base_folder = config.section + "_" + config.mode + "_" + config.target[
                                                             1:-1] + "_header=" + config.section_header + "_rltv=" + str(
        config.rltv) + "_section_emb=" + str(config.section_emb) + "_section_avg_sep=" + config.section_avg_sep + \
                  "_augmentation_mode=" + str(config.augmentation_mode) + "_header_information_contextual_" + str(
        config.header_information_contextual)

    checkpoint_path = os.path.join(base_folder, config.checkpoint)
    all_subdirectories = os.listdir(checkpoint_path)
    all_subdirectories.sort()
    micro_auc_all = []
    micro_precision_all = []
    micro_recall_all = []
    micro_f1_all = []
    macro_precision_all = []
    macro_recall_all = []
    macro_f1_all = []

    for subdirectories in all_subdirectories:

        new_path = os.path.join(checkpoint_path, subdirectories)

        if config.mode == "contextual":
            test_dataset, list_name = contextual_load(config.train_file, config)
            test = test_dataset
        else:
            test_dataset, list_name = data_load(config.train_file, test_pmids, config, folder_no, current_mode="test")
            test = test_dataset

        num_labels = len(list_name[0])

        model = BERT(config, num_labels)
        model.load_bert(config.bert_model_name)
        path_to_model = os.path.join(new_path, "best.mdl")
        model.load_state_dict(torch.load(path_to_model)['model'])

        epoch = ""

        test_batch_num = len(test) // config.eval_batch_size + (len(test) % config.eval_batch_size != 0)

        if config.use_gpu:
            model.cuda()

        Loss = torch.nn.BCELoss()

        sid, report, running_loss, target_result, valid_result, micro_auc = evaluate(model, Loss, test,config,test_batch_num,epoch,'TEST',list_name, base_folder_name=base_folder)

        print("micro_auc: ", micro_auc)
        micro_auc_all.append(micro_auc)
        micro_precision_all.append(report['micro avg']['precision'])
        micro_recall_all.append(report['micro avg']['recall'])
        micro_f1_all.append(report['micro avg']['f1-score'])
        macro_precision_item = []
        macro_recall_item = []
        macro_f1_item = []
        if config.test_on_specific_section == "Methods":
            target_names = ["3a", "3b", "4a", "4b", "5", "6a", "6b", "7a", "7b", "8a", "8b", "9", "10", "11a", "11b",
                            "12a", "12b"]
        elif config.test_on_specific_section == "Results":
            target_names = ["13a", "13b", "14a", "14b", "15", "16", "17a", "17b", "18", "19"]
        elif config.test_on_specific_section == "Discussion":
            target_names = ["20", "21", "22"]
        else:
            target_names = ['2b', '3a', '3b', '4a', '4b', '5', '6a', '6b', '7a', '7b', '8a', '8b', '9', '10',
                            '11a', '11b', '12a', '12b', '13a', '13b', '14a', '14b', '15', '16', '17a', '17b', '18',
                            '19', '20',
                            '21', '22', '23', '24', '25']
        for target_name in target_names:
            macro_precision_item.append(report[target_name]["precision"])
            macro_recall_item.append(report[target_name]["recall"])
            macro_f1_item.append(report[target_name]["f1-score"])
        macro_precision_all.append(sum(macro_precision_item) / len(macro_precision_item))
        macro_recall_all.append(sum(macro_recall_item) / len(macro_recall_item))
        macro_f1_all.append(sum(macro_f1_item) / len(macro_f1_item))

        df = pd.DataFrame(list(zip(sid, target_result, valid_result)), columns=["sid", "target_result", "valid_result"])

        df.to_csv(os.path.join(new_path, "test_predictions.csv"))

        # folder_no += 1

    print("micro_auc_all: ", micro_auc_all)
    print("micro_precision_all: ", micro_precision_all)
    print("macro_precision_all: ", macro_precision_all)
    print("micro_recall_all: ", micro_recall_all)
    print("macro_recall_all: ", macro_recall_all)
    print("micro_f1_all: ", micro_f1_all)
    print("macro_f1_all: ", macro_f1_all)

    print("micro precision: ")
    print((sum(micro_precision_all) / len(micro_precision_all), st.tstd(micro_precision_all)))
    print("macro precision: ")
    print((sum(macro_precision_all) / len(macro_precision_all), st.tstd(macro_precision_all)))

    print("micro recall: ")
    print((sum(micro_recall_all) / len(micro_recall_all), st.tstd(micro_recall_all)))
    print("macro recall: ")
    print((sum(macro_recall_all) / len(macro_recall_all), st.tstd(macro_recall_all)))

    print("micro F1: ")
    print((sum(micro_f1_all) / len(micro_f1_all), st.tstd(micro_f1_all)))
    print("macro F1: ")
    print((sum(macro_f1_all) / len(macro_f1_all), st.tstd(macro_f1_all)))

    print("micro auc: ")
    print((sum(micro_auc_all) / len(micro_auc_all), st.tstd(micro_auc_all)))

if __name__ == '__main__':

    # hyperparameters
    parser = ArgumentParser()

    parser.add_argument('--train_file', type=str,
                        help='path to the training file')

    parser.add_argument('--test_file', type=str,
                        help='path to the test file')

    parser.add_argument('--use_gpu', type=bool, default=1,
                        help='if or not use GPU, choose from True or False')

    parser.add_argument('--gpu_device', type=int, default=0, help='number of GPU devices')

    parser.add_argument('--target', type=str,
                        help='target token, choose either [CLS] or [SEP]')

    parser.add_argument('--mode', type=str,
                        help='input mode, choose either contextual or single')

    parser.add_argument('--save', type=int, default=1,
                        help='create log file or not, choose either 1 or 0')

    parser.add_argument('--section_header', type=str,
                        help='section header to involve, choose from none, both, inner or outer')

    parser.add_argument('--bert_model_name', type=str,
                        help='pretrained language model name, choose from huggingface')

    parser.add_argument('--bert_dropout', type=float, default=0.1,
                        help='dropout rate')

    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch size')

    parser.add_argument('--eval_batch_size', type=int, default=4,
                        help='batch size')

    parser.add_argument('--max_epoch', type=int, default=20,
                        help='number of epoch')

    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay rate of linear neural network')

    parser.add_argument('--bert_weight_decay', type=float, default=0,
                        help='weight decay rate of the bert model')

    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='learning rate of linear neural network')

    parser.add_argument('--bert_learning_rate', type=float, default=1e-5,
                        help='learning rate of bert finetuning')

    parser.add_argument('--early_stopping', type=int, default=0,
                        help='early stopping')

    parser.add_argument('--lr_scheduler', type=int, default=0,
                        help='learning rate scheduler')

    parser.add_argument('--header_information_contextual', type=int,
                        help='0 - not involve any header information in contextual setting; 1 - involve header information for the current sentence only in contextual \
                        setting; 2 - involve header information for the preceding, current and trailing sentences in contextual setting')

    parser.add_argument('--augmentation_mode', type=int, help='0 - no augmentation; 1 - (generative by GPT-4) concatenate all augmentation outcomes with the \
                        original samples in the training set;  2 - (generative by GPT-4) split augmentation outcomes into different folders in the training set; \
                        3 - (rephrasing by GPT-4) add the rephrased data to the training set; \
                        4 - (EDA) add the data augmented by EDA method; \
                        5 - (UMLS-EDA) add the data augmented by UMLS-EDA method')

    parser.add_argument('--augmentation_file', type=str, help='the path to the augmentation file')

    parser.add_argument('--sent_dim', type=int,
                        help='dimension of relative/absolute sentence position embedding')

    parser.add_argument('--rltv', type=int,
                        help='use relative/absolute/none sentence position (1/0/-1)')

    parser.add_argument('--section_emb', type=int,
                        help='whether the model would add section headers as a separate feature')

    parser.add_argument('--position_emb', type=int,
                        help='whether the model would add position embedding')

    parser.add_argument('--section_dim', type=int,
                        help='the number of dimensions of section headers')

    parser.add_argument('--rltv_bins_num', type=int,
                        help='number of bins for relative postion embedding')

    parser.add_argument('--section_avg_sep', type=str,
                        help='whether the model would add section headers as a separate feature - use seperate method (sep) or average (avg)')

    parser.add_argument('--section', type=str,
                        help='select the section-specific data from whole/Methods/Results/Discussion and train the model')

    parser.add_argument('--checkpoint', type=str,
                        help='location of the checkpoint file')

    parser.add_argument('--folder_no', type=int,
                        help='select data from a specific folder (choose from 1/2/3/4/5)')

    parser.add_argument('--test_on_specific_section', type=str,
                        help='test a model performance on which specific section? choose from Methods, Results, Discussion or whole')

    config = parser.parse_args()
    print("config", config)
    test(config)
