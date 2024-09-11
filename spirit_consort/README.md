# SPIRIT-CONSORT

This repository is broken into two key sections: data and src. 'data' contains all the data in CSV format - 100 pairs of randomized controlled trial (RCT) protocols results articles annotated with fine-grained SPIRIT and CONSORT checklist items at term level, sentence level, and article level. 'src' contains relevant code for training and evaluating the baseline model (PubMedBERT) on multi-label, multi-instance classification tasks at the sentence, section, and article bag-levels. Additionally, there are custom "1+" metrics that only consider bags (i.e., sections or articles) as being correct if 1 or more sentences are correctly predicted.

Before running anything, be sure to install all required packages in src/requirements.txt and adjust src/models/config.json as necessary. This will most likely just involve changing your log_path. After that, training the model should be as simple as running the following command in the src directory:

```console
python train.py
```
