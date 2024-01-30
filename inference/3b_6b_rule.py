import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics import f1_score
from argparse import ArgumentParser
import pandas as pd
from ast import literal_eval
import requests
import re
import ast

nltk.download('punkt')
porter = PorterStemmer()
lancaster=LancasterStemmer()



def stem_sentence(sentence):
	tokenized_words=word_tokenize(sentence)
	tokenized_sentence = []
	for word in tokenized_words:
		tokenized_sentence.append(porter.stem(word))
	tokenized_sentence = " ".join(tokenized_sentence)
	return tokenized_sentence

def check_3b(l1):
	for i in l1:
		if i == "3b":
			return True
	return False

def check_3b_content(sentence, phrases_3b):
	stemmed_sentence = stem_sentence(sentence)
	for phrase_3b in phrases_3b:
		if phrase_3b in stemmed_sentence:
			print("3b stemmed_sentence: ", sentence)
			print("phrase_3b: ", phrase_3b)
			return True
	return False

def check_6b(l1):
	for i in l1:
		if i == "6b":
			return True
	return False

def check_6b_content(sentence, phrases_6b):
	stemmed_sentence = stem_sentence(sentence)
	for phrase_6b in phrases_6b:
		if phrase_6b in stemmed_sentence:
			print("6b stemmed_sentence: ", sentence)
			print("phrase_6b: ", phrase_6b)
			return True
	return False

def main():
	parser = ArgumentParser()

	parser.add_argument('--file_path', default='all_CONSORT_manual_data.csv', type=str, help='path to the file')

	opt = parser.parse_args()
	file_path = opt.file_path

	df = pd.read_csv(file_path)

	df["CONSORT_Item"] = df["CONSORT_Item"].apply(ast.literal_eval)
	df["section"] = df["section"].apply(ast.literal_eval)

	phrases_3b = [stem_sentence(i) for i in ["no longer feasible","decision was taken", \
	"decision was made","committee agreed"]]
	phrases_6b = [stem_sentence(i) for i in ["original trial protocol","trial because of", \
	"early termination","the trial because","the original trial"]]

	print("phrases_3b: ", phrases_3b)
	print("phrases_6b: ", phrases_6b)

	df["3bs"] = df["CONSORT_Item"].apply(check_3b)
	df["3b_check"] = df.apply(lambda x: check_3b_content(x['text'], phrases_3b), axis=1)
	df["6bs"] = df["CONSORT_Item"].apply(check_6b)
	df["6b_check"] = df.apply(lambda x: check_6b_content(x['text'], phrases_6b), axis=1)
	
	print("3b f1: ", f1_score(df[df["3bs"]==True]["3bs"].to_list(), df[df["3bs"]==True]["3b_check"].to_list()))
	print("6b f1: ", f1_score(df[df["6bs"]==True]["6bs"].to_list(), df[df["6bs"]==True]["6b_check"].to_list()))

if __name__ == "__main__":
	main()