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
nltk.download('punkt')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from ast import literal_eval as load

ps = PorterStemmer()

def check_1b(l1):
	for i in l1:
		if i == "1b":
			return True

def check_title(l1):
	if l1.lower() in ["title", "titles"]:
		return True
	return False

def check_1a_title(text):
	words = word_tokenize(text)
	for w in words:
		w = w.split("-")
		for i in w:
			if ps.stem(i.lower()) in ["random", "randomis"]:
				return True
	return False


def check_1a(l1):
	for i in l1:
		if i == "1a":
			return True

def check_start(string, sample) :
# check if string starts with the substring

	string = string.lower()
	sample = sample.lower()
	if (sample in string):
		y = "^" + sample
		x = re.search(y, string)
		if x :
			return True
		else :
			return False
	else :
			return False

def check_1b_by_map(str1, structured_abstract_items):
	for i in structured_abstract_items:
		if check_start(str1, i):
			return True
	return False

def find_1b_candidates(str1):
	if str1.lower() in ['summary', 'abstract']:
		return True
	return False

def check_1a_1b_with_pmids(pmids):

	file_path = "data/all_CONSORT_manual_data.csv"
	df = pd.read_csv(file_path)
	df["CONSORT_Item"] = df["CONSORT_Item"].apply(ast.literal_eval)
	df["section"] = df["section"].apply(ast.literal_eval)
	df = df[df["PMCID"].isin(pmids)]

# check 1a
	ps = PorterStemmer()
	df["1as"] = df["CONSORT_Item"].apply(check_1a)
	df_1as = df[df["1as"] == True]

	df["title_check"] = df["section.1"].apply(check_title)
	df_title_check = df[df["title_check"] == True]
	df_title_check["1a_check"] = df_title_check["text"].apply(check_1a_title)
	df_title_check = df_title_check[df_title_check["1a_check"] == True]

	PMCID_list_1a = set(df_title_check.PMCID.to_list())
	pmids_with_1a= set(df_1as.PMCID.to_list())
	pmids_without_1a = set(df.PMCID.to_list()) - set(df_1as.PMCID.to_list())

	false_positives_1a = PMCID_list_1a - pmids_with_1a
	false_negatives_1a = pmids_with_1a - PMCID_list_1a
	true_negatives_1a = pmids_without_1a - PMCID_list_1a
	true_positives_1a = PMCID_list_1a - pmids_without_1a

	precision_1a = len(true_positives_1a) / (len(true_positives_1a) + len(false_positives_1a))
	recall_1a = len(true_positives_1a) / (len(true_positives_1a) + len(false_negatives_1a))
	f1_score_1a = 2 * (precision_1a * recall_1a) / (precision_1a + recall_1a)

# check 1b
	df["1b_check"] = df["CONSORT_Item"].apply(check_1b)

	pmids_with_1b = set(df[df["1b_check"] == True].PMCID.to_list())
	pmids_without_1b = set(df.PMCID.to_list()) - set(df[df["1b_check"] == True].PMCID.to_list())

	# load the NLM structured abstract labels 

	response = requests.get("https://lhncbc.nlm.nih.gov/ii/areas/structured-abstracts/downloads/Structured-Abstracts-Labels-102615.txt")
	data = response.text
	structured_abstract_items = []

	for i in data.split("\n"):
		structured_abstract_items.append(i.split("|")[0])
	structured_abstract_items.remove("")

	df["1b_candidates"] = df["section.1"].apply(find_1b_candidates)
	df_1b_candidates = df[df["1b_candidates"] == True]

	# check the start of the abstract

	first_or_second = []

	previous_pmcid = ""
	is_first = False

	for key, i in df_1b_candidates.iterrows():
		if i.PMCID != previous_pmcid:
			first_or_second.append(True)
			is_first = True
			previous_pmcid = i.PMCID
		else:
			if is_first:
				first_or_second.append(True)
				is_first = False
			else:
				first_or_second.append(False)

	df_1b_candidates["first_or_second"] = first_or_second
	df_1b_candidates = df_1b_candidates[df_1b_candidates["first_or_second"] == True]

	df_1b_candidates['text'] = df_1b_candidates['text'].str.lower()

	PMCID_list_1b = set(df_1b_candidates[df_1b_candidates.apply(lambda row: check_1b_by_map(row['text'], structured_abstract_items), axis=1) == True].PMCID.to_list())
	
	false_positives_1b = PMCID_list_1b - set(df[df["1b_check"] == True].PMCID.to_list())
	false_negatives_1b = set(pmids_with_1b) - PMCID_list_1b
	true_negatives_1b = set(pmids_without_1b) - PMCID_list_1b
	true_positives_1b = PMCID_list_1b - set(pmids_without_1b)

	precision_1b = len(true_positives_1b) / (len(true_positives_1b) + len(false_positives_1b))
	recall_1b = len(true_positives_1b) / (len(true_positives_1b) + len(false_negatives_1b))
	f1_score_1b = 2 * (precision_1b * recall_1b) / (precision_1b + recall_1b)

	correct_numbers_1a_1b = len(true_positives_1b) + len(true_positives_1a)
	pred_numbers_1a_1b =  len(true_positives_1b) + len(true_positives_1a) + len(false_positives_1b) + len(false_positives_1a)
	gold_numbers_1a_1b = len(false_negatives_1b) + len(true_positives_1b) + len(false_negatives_1a) + len(true_positives_1a)

	dict_1a_1b = {'1a': {'prec': precision_1a, 'rec': recall_1a, 'f1': f1_score_1a}, '23': {'prec': precision_1b, 'rec': recall_1b, 'f1': f1_score_1b}}

	return correct_numbers_1a_1b, pred_numbers_1a_1b, gold_numbers_1a_1b, dict_1a_1b

def save_div(a, b):
	if b != 0:
		return a / b
	else:
		return 0.0


def evaluation(gold_labels, pred_labels, correct_numbers_1a_1b, pred_numbers_1a_1b, gold_numbers_1a_1b):
	result = {}
	for label in label_name:
		result[label] = {"prec": 0.0, "rec": 0.0, "f1": 0.0, "support": 0}

	total_pred_num, total_gold_num, total_correct_num = 0.0, 0.0, 0.0

	for i in range(len(gold_labels)):

		pred_labels_i = pred_labels[i]
		gold_labels_i = gold_labels[i]

		for idx in gold_labels_i:
			result[idx]["support"] += 1
			total_gold_num += 1
			result[idx]["rec"] += 1

		for idx in pred_labels_i:
			total_pred_num += 1
			result[idx]["prec"] += 1

			if idx in gold_labels_i:
				total_correct_num += 1
				result[idx]["f1"] += 1

		total_correct_num += correct_numbers_1a_1b
		total_pred_num += pred_numbers_1a_1b
		total_gold_num += gold_numbers_1a_1b

	for label in result:
		counts = result[label]
		counts["prec"] = save_div(counts["f1"], counts["prec"])
		counts["rec"] = save_div(counts["f1"], counts["rec"])
		counts["f1"] = save_div(2*counts["prec"]*counts["rec"], counts["prec"]+counts["rec"])

	item_precs = []
	item_recalls = []
	item_f1s = []

	for label in result:
		item_precs.append(result[label]["prec"])
		item_recalls.append(result[label]["rec"])
		item_f1s.append(result[label]["f1"])

	micro_prec = save_div(total_correct_num , total_pred_num)
	micro_rec = save_div(total_correct_num, total_gold_num)
	micro_f1 = save_div(2*micro_prec*micro_rec, micro_prec+micro_rec)

	macro_prec = sum(item_precs) / len(item_precs)
	macro_rec = sum(item_recalls) / len(item_recalls)
	macro_f1 = sum(item_f1s) / len(item_f1s)

	return micro_prec, micro_rec, micro_f1, macro_prec, macro_rec, macro_f1, result


if __name__ == '__main__':
	path = "whole_contextual_CLS_header=both_rltv=-1_section_emb=0_section_avg_sep=_augmentation_mode=0_header_information_contextual_2"

	dir_list = os.listdir(path)
	base = dir_list[0]
	base_path = os.path.join(path, base)

	dir_list_folders = os.listdir(base_path)

	section = "whole"

	if section == "Methods":
		label_name = ["3a", "3b", "4a", "4b", "5", "6a", "6b", "7a", "7b", "8a", "8b", "9", "10", "11a", "11b", "12a", "12b"]
	elif section == "Results":
		label_name = ["13a", "13b", "14a", "14b", "15", "16", "17a", "17b", "18", "19"]
	elif section == "Discussion":
		label_name = ["20", "21", "22"]
	else:
		label_name = ['2b', '3a', '3b', '4a', '4b', '5', '6a', '6b', '7a', '7b', '8a', '8b', '9', '10',
				  '11a', '11b', '12a', '12b', '13a', '13b', '14a', '14b', '15', '16', '17a', '17b', '18', '19', '20',
				  '21', '22', '23', '24', '25']

	precs, recs, f1s, macro_precs, macro_recs, macro_f1s, results = [], [], [], [], [], [], []


	for folder in dir_list_folders:
		pmids = []
		trun_labels_for_pmid = []
		predicted_labels_for_pmid = []
		new_base = os.path.join(base_path, folder, "test_predictions.csv")
		new_predictions = pd.read_csv(new_base, dtype=str)
		new_predictions["target_result"]= new_predictions["target_result"].apply(load)
		new_predictions["valid_result"]= new_predictions["valid_result"].apply(load)

		new_predictions['pmid'] = new_predictions['sid'].str.split('_').str[0]

		new_pmids = new_predictions.pmid.to_list()
		new_pmids = list(set(new_pmids))
		for new_pmid in new_pmids:
			true_labels = new_predictions[new_predictions["pmid"] == new_pmid].target_result.to_list()
			predicted_labels = new_predictions[new_predictions["pmid"] == new_pmid].valid_result.to_list()
			true_labels = list(set([i for j in true_labels for i in j]))
			predicted_labels = list(set([i for j in predicted_labels for i in j]))
			true_labels.remove('0')
			predicted_labels.remove('0')

			pmids.append(new_pmid)
			trun_labels_for_pmid.append(true_labels)
			predicted_labels_for_pmid.append(predicted_labels)

		correct_numbers_1a_1b, pred_numbers_1a_1b, gold_numbers_1a_1b, dict_1a_1b = check_1a_1b_with_pmids(pmids)
		micro_prec, micro_rec, micro_f1, macro_prec, macro_rec, macro_f1, result = evaluation(trun_labels_for_pmid, predicted_labels_for_pmid, correct_numbers_1a_1b, pred_numbers_1a_1b, gold_numbers_1a_1b)
		precs.append(micro_prec)
		recs.append(micro_rec)
		f1s.append(micro_f1)
		macro_precs.append(macro_prec)
		macro_recs.append(macro_rec)
		macro_f1s.append(macro_f1)

		results.append(result)

	prec_mean_std = (sum(precs) / len(precs), st.tstd(precs))
	recs_mean_std = (sum(recs) / len(recs), st.tstd(recs))
	f1s_mean_std = (sum(f1s) / len(f1s), st.tstd(f1s))
	macro_prec_mean_std = (sum(macro_precs) / len(macro_precs), st.tstd(macro_precs))
	macro_recs_mean_std = (sum(macro_recs) / len(macro_recs), st.tstd(macro_recs))
	macro_f1s_mean_std = (sum(macro_f1s) / len(macro_f1s), st.tstd(macro_f1s))

	print("prec_mean_std: ", prec_mean_std)
	print("recs_mean_std: ", recs_mean_std)
	print("f1s_mean_std: ", f1s_mean_std)
	print("macro_prec_mean_std: ", macro_prec_mean_std)
	print("macro_recs_mean_std: ", macro_recs_mean_std)
	print("macro_f1s_mean_std: ", macro_f1s_mean_std)

	results_mean_std = {}

	for label in label_name:
		results_mean_std[label] = {}
		for item in ['prec', 'rec', 'f1']:
			results_mean_std[label][item] = []
			for folder_no in range(len(results)):
				results_mean_std[label][item].append(results[folder_no][label][item])

	for label in label_name:
		for item in ['prec', 'rec', 'f1']:
			scores_across_folder = results_mean_std[label][item]
			results_mean_std[label][item] = (sum(scores_across_folder) / len(scores_across_folder), st.tstd(scores_across_folder))

	print(results_mean_std)







