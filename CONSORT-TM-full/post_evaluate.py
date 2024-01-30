import pandas as pd
import os, json
import scipy.stats as st
from argparse import ArgumentParser

def agg_folders(config):

    label_name = ['2b', '3a', '3b', '4a', '4b', '5', '6a', '6b', '7a', '7b', '8a', '8b', '9', '10',
                  '11a', '11b', '12a', '12b', '13a', '13b', '14a', '14b', '15', '16', '17a', '17b', '18', '19', '20',
                  '21', '22', '23', '24', '25', 'micro', 'macro', 'samples', 'weighted']

    folds = sorted([f for f in os.listdir(config.path)])
    precision, recall, f1, support = dict([(k, []) for k in label_name]), dict([(k, []) for k in label_name]), dict([(k, []) for k in label_name]), dict([(k, []) for k in label_name])
    for fold in folds:
        log = open(os.path.join(config.path, fold,'log.txt'), "r").readlines()
        best_n = json.loads(log[-1])['best epoch']
        info = json.loads(log[best_n+1])['report_test'].split('\n')[1:]
        for_macro_precision = []
        for_macro_recall = []
        for_macro_f1 = []
        for line in info:
            row = line.split()

            if row != []:
                item = row[0]
                if item not in label_name:
                    continue

                if item != "macro":
                    precision[item].append(float(row[-4]))
                    recall[item].append(float(row[-3]))
                    f1[item].append(float(row[-2]))
                    support[item].append((float(row[-1])))
                    if item not in ['micro', 'samples', 'weighted']:
                        for_macro_precision.append(float(row[-4]))
                        for_macro_recall.append(float(row[-3]))
                        for_macro_f1.append(float(row[-2]))

        precision["macro"].append(sum(for_macro_precision) / len(for_macro_precision))
        recall["macro"].append(sum(for_macro_recall) / len(for_macro_recall))
        f1["macro"].append(sum(for_macro_f1) / len(for_macro_f1))

    report = pd.DataFrame(columns=['precision', 'recall', 'f1'])
    for label in label_name:
        report = report._append(pd.Series({'precision': precision[label], 'recall': recall[label], 'f1': f1[label]}, name=label))
    print("report: ", report)

    print("Average score: ")
    print(report.applymap(lambda x: sum(x)/len(x)))
    avg_report = report.applymap(lambda x: sum(x)/len(x))
    print("section=" + config.section + "_results_" + config.path.split("/")[0] + "_avg.csv")
    avg_report.to_csv("section=" + config.section + "_results_" + config.path.split("/")[0] + "_avg.csv")

    for i in range(report.shape[0]):
        for j in range(report.shape[1]):
            value = report.iloc[i, j]
            report.iloc[i, j] = (sum(value) / len(value), st.tstd(value))
    print("Boost strap confidence interval: ")
    print(report)

    if "/" in config.path:
        report.to_csv("section=" + config.section + "_results_" + config.path.split("/")[0] + "_ci95.csv")
    elif "\\" in config.path:
        report.to_csv("section=" + config.section + "_results_" + config.path.split("\\")[0] + "_ci95.csv")


if __name__ == '__main__':

    # hyperparameters
    parser = ArgumentParser()
    parser.add_argument('--path', type=str,
                        help='path to the checkpoints folder')
    parser.add_argument('--section', type=str,
                        help='test on the data from which section - choose from Methods/Results/Discussion/Whole')
    config = parser.parse_args()
    print("config", config)

    agg_folders(config)