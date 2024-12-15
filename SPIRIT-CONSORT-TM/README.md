# SPIRIT-CONSORT-TM

This repository is broken into three key sections: data, src_sentence_level, and src_term_level. 

* The folder **data** contains all the data in CSV format - 100 pairs of randomized controlled trial (RCT) protocols results articles annotated with fine-grained SPIRIT and CONSORT checklist items at term level, sentence level, and article level. 

* The folder **src_sentence_level** contains relevant code for training and evaluating the baseline model (PubMedBERT) on multi-label, multi-instance classification tasks at the sentence, section, and article bag-levels. Additionally, there are custom "1+" metrics that only consider bags (i.e., sections or articles) as being correct if 1 or more sentences are correctly predicted.

* The folder **src_term_level** contains code of the training and evaluation process of our term extraction task.

For data, code, and materials related to the searching and processing of PubMed search results and screening of articles that generated training data, please see here: https://osf.io/8rg4h/

