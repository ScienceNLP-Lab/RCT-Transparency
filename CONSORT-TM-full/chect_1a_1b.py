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

def main():
	# parser = ArgumentParser()
	#
	# parser.add_argument('--file_path', type=str, help='path to the file')
	#
	# opt = parser.parse_args()
	file_path = 'data/all_CONSORT_manual_data.csv'

	df = pd.read_csv(file_path)

	df["CONSORT_Item"] = df["CONSORT_Item"].apply(ast.literal_eval)
	df["section"] = df["section"].apply(ast.literal_eval)

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

	precision = len(true_positives_1a) / (len(true_positives_1a) + len(false_positives_1a))
	recall = len(true_positives_1a) / (len(true_positives_1a) + len(false_negatives_1a))
	f1_score = 2 * (precision * recall) / (precision + recall)


	print("____________________________________")
	print("results for 1a --- ")
	print("true_positives_1a: ", true_positives_1a)
	print("number of true positives: ", len(true_positives_1a))
	print("true_negatives_1a: ", true_negatives_1a)
	print("number of true negatives: ", len(true_negatives_1a))
	print("false_positives_1a: ", false_positives_1a)
	print("number of false positives: ", len(false_positives_1a))
	print("false_negatives_1a: ", false_negatives_1a)
	print("number of false negatives: ", len(false_negatives_1a))

	print("")
	print("precision: ", precision)
	print("recall: ", recall)
	print("f1: ", f1_score)

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

	precision = len(true_positives_1b) / (len(true_positives_1b) + len(false_positives_1b))
	recall = len(true_positives_1b) / (len(true_positives_1b) + len(false_negatives_1b))
	f1_score = 2 * (precision * recall) / (precision + recall)


	print("____________________________________")
	print("results for 1b --- ")
	print("true_positives_1b: ", true_positives_1b)
	print("number of true positives: ", len(true_positives_1b))
	print("true_negatives_1b: ", true_negatives_1b)
	print("number of true negatives: ", len(true_negatives_1b))
	print("false_positives_1b: ", false_positives_1b)
	print("number of false positives: ", len(false_positives_1b))
	print("false_negatives_1b: ", false_negatives_1b)
	print("number of false negatives: ", len(false_negatives_1b))

	print("")
	print("precision: ", precision)
	print("recall: ", recall)
	print("f1: ", f1_score)


if __name__ == "__main__":
	main()