import pandas as pd
import os
import scipy.stats as st
from argparse import ArgumentParser
import pandas as pd
from ast import literal_eval
import requests
import re
import ast
import nltk
import numpy as np
from ast import literal_eval as load
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn
import matplotlib.pyplot as plt


def plot_confusion_matrix(data, labels, output_filename):
    """Plot confusion matrix using heatmap.

    Args:
        data (list of list): List of lists with confusion matrix data.
        labels (list): Labels which will be plotted across x and y axis.
        output_filename (str): Path to output file.

    """
    seaborn.set(color_codes=True)
    plt.figure(1, figsize=(15, 15))
    plt.title("Confusion Matrix")
    seaborn.set(font_scale=1)
    ax = seaborn.heatmap(data, annot=data, cmap="YlGnBu", fmt='g')
    print("labels: ", labels)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set(ylabel="True Label", xlabel="Predicted Label")
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def plot_multi_label_cm(true, pred, num_label):
    n = num_label

    # Create an n x n matrix filled with zeros using NumPy
    cm = np.zeros((n-1, n-1))

    for i in range(len(true)):
        for j in pred[i]:
            for k in true[i]:
                if k == 0 or j == 0:
                    continue
                else:
                    cm[k-1][j-1] += 1
    return cm

def convert_label_to_id(l1, labels_to_id):
    returned_list = []
    for i in l1:
        returned_list.append([labels_to_id[j] for j in i])
    return returned_list


if __name__ == '__main__':
    path = "whole_contextual_CLS_header=both_rltv=-1_section_emb=0_section_avg_sep=_augmentation_mode=0_header_information_contextual_2"

    dir_list = os.listdir(path)
    base = dir_list[0]
    base_path = os.path.join(path, base)

    dir_list_folders = os.listdir(base_path)


    label_name = ['0', '2b', '3a', '3b', '4a', '4b', '5', '6a', '6b', '7a', '7b', '8a', '8b', '9', '10',
                  '11a', '11b', '12a', '12b', '13a', '13b', '14a', '14b', '15', '16', '17a', '17b', '18', '19',
                  '20', '21', '22', '23', '24', '25']
    label_to_id = {}

    for index, element in enumerate(label_name):
        label_to_id[element] = index
    print(label_to_id)
    precs, recs, f1s, macro_precs, macro_recs, macro_f1s, results = [], [], [], [], [], [], []
    all_true_labels = []
    all_predicted_labels = []
    for folder in dir_list_folders:
        pmids = []
        trun_labels_for_pmid = []
        predicted_labels_for_pmid = []
        new_base = os.path.join(base_path, folder, "test_predictions.csv")
        new_predictions = pd.read_csv(new_base, dtype=str)
        new_predictions["target_result"] = new_predictions["target_result"].apply(load)
        new_predictions["valid_result"] = new_predictions["valid_result"].apply(load)

        new_predictions['pmid'] = new_predictions['sid'].str.split('_').str[0]

        true_labels = new_predictions.target_result.to_list()
        predicted_labels = new_predictions.valid_result.to_list()

        all_true_labels.extend(true_labels)
        all_predicted_labels.extend(predicted_labels)

    all_true_labels_id = convert_label_to_id(all_true_labels, label_to_id)
    all_predicted_labels_id = convert_label_to_id(all_predicted_labels, label_to_id)
    cm = plot_multi_label_cm(all_true_labels_id, all_predicted_labels_id, len(label_name))
    plot_confusion_matrix(cm, label_name[1:], "confusion_matrix.png")
