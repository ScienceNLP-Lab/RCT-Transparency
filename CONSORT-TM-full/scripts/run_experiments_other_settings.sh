# current sentence content + Section headers (prepended to current sentence) + Contextual information + sentence representation [CLS]
python train.py --target="[CLS]" --section_header="both" --bert_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --augmentation_mode=0 --sent_dim=0 --rltv=-1 --section_emb=0 --position_emb=0 --section_dim=0 --train_file="data/all_CONSORT_manual_data.csv" --test_file="" --mode="contextual" --section_avg_sep="" --section="whole" --header_information_contextual=1

# current sentence content + Contextual information + sentence representation [CLS]
python train.py --target="[CLS]" --section_header="both" --bert_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --augmentation_mode=0 --sent_dim=0 --rltv=-1 --section_emb=0 --position_emb=0 --section_dim=0 --train_file="data/all_CONSORT_manual_data.csv" --test_file="" --mode="contextual" --section_avg_sep="" --section="whole" --header_information_contextual=0

# current sentence content + Section headers (prepended to preceding, current and trialing sentences) + Contextual information & sentence representation [SEP]
python train.py --target="[SEP]" --section_header="both" --bert_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --augmentation_mode=0 --sent_dim=0 --rltv=-1 --section_emb=0 --position_emb=0 --section_dim=0 --train_file="data/all_CONSORT_manual_data.csv" --test_file="" --mode="contextual" --section_avg_sep="" --section="whole" --header_information_contextual=2

# current sentence content + Section headers (prepended) & Inner
python train.py --target="[CLS]" --section_header="inner" --bert_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --augmentation_mode=0 --sent_dim=0 --rltv=-1 --section_emb=0 --position_emb=0 --section_dim=0 --train_file="data/all_CONSORT_manual_data.csv" --mode="single" --section_avg_sep="" --section="whole" --rltv_bins_num=0

# current sentence content + Section headers (prepended) & Outer
python train.py --target="[CLS]" --section_header="outer" --bert_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --augmentation_mode=0 --sent_dim=0 --rltv=-1 --section_emb=0 --position_emb=0 --section_dim=0 --train_file="data/all_CONSORT_manual_data.csv" --mode="single" --section_avg_sep="" --section="whole" --rltv_bins_num=0

# current sentence content + Section headers (prepended) & All
python train.py --target="[CLS]" --section_header="both" --bert_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --augmentation_mode=0 --sent_dim=0 --rltv=-1 --section_emb=0 --position_emb=0 --section_dim=0 --train_file="data/all_CONSORT_manual_data.csv" --mode="single" --section_avg_sep="" --section="whole" --rltv_bins_num=0

# vanilla PubMedBERT sentence classification
# current sentence content
python train.py --target="[CLS]" --section_header="none" --bert_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --augmentation_mode=0 --sent_dim=0 --rltv=-1 --section_emb=0 --position_emb=0 --section_dim=0 --train_file="data/all_CONSORT_manual_data.csv" --mode="single" --section_avg_sep="" --section="whole" --rltv_bins_num=0

# current sentence content + Section headers (seperately encoded) & [CLS]
python train.py --target="[CLS]" --section_header="both" --bert_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --augmentation_mode=0 --sent_dim=0 --rltv=-1 --section_emb=1 --position_emb=0 --section_dim=300 --train_file="data/all_CONSORT_manual_data.csv" --mode="single" --section_avg_sep="sep" --section="whole" --rltv_bins_num=0

# current sentence content + Section headers (seperately encoded) & average
python train.py --target="[CLS]" --section_header="both" --bert_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --augmentation_mode=0 --sent_dim=0 --rltv=-1 --section_emb=1 --position_emb=0 --section_dim=300 --train_file="data/all_CONSORT_manual_data.csv" --mode="single" --section_avg_sep="avg" --section="whole" --rltv_bins_num=0

# current sentence content + Section headers (prepended) + Sentence position & absolute
python train.py --target="[CLS]" --section_header="both" --bert_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --augmentation_mode=0 --sent_dim=50 --rltv=0 --section_emb=0 --position_emb=1 --section_dim=0 --train_file="data/all_CONSORT_manual_data.csv" --mode="single" --section_avg_sep="" --rltv_bins_num=10 --section="whole" --rltv_bins_num=0

# current sentence content + Section headers (prepended) + Sentence position & relative
python train.py --target="[CLS]" --section_header="both" --bert_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --augmentation_mode=0 --sent_dim=50 --rltv=1 --section_emb=0 --position_emb=1 --section_dim=0 --train_file="data/all_CONSORT_manual_data.csv" --mode="single" --section_avg_sep="" --rltv_bins_num=10 --section="whole" --rltv_bins_num=0

# Best model trained with data from specific sections
# Best model setting: current sentence content + Section headers (prepended to preceding, current and trialing sentences) + Contextual information + sentence representation [CLS]
# Methods
python train.py --target="[CLS]" --section_header="both" --bert_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --augmentation_mode=0 --sent_dim=0 --rltv=-1 --section_emb=0 --position_emb=0 --section_dim=0 --train_file="data/all_CONSORT_manual_data.csv" --mode="contextual" --section_avg_sep="" --section="Methods" --header_information_contextual=2

# Results
python train.py --target="[CLS]" --section_header="both" --bert_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --augmentation_mode=0 --sent_dim=0 --rltv=-1 --section_emb=0 --position_emb=0 --section_dim=0 --train_file="data/all_CONSORT_manual_data.csv" --mode="contextual" --section_avg_sep="" --section="Results" --header_information_contextual=2

# Discussion
python train.py --target="[CLS]" --section_header="both" --bert_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --augmentation_mode=0 --sent_dim=0 --rltv=-1 --section_emb=0 --position_emb=0 --section_dim=0 --train_file="data/all_CONSORT_manual_data.csv" --mode="contextual" --section_avg_sep="" --section="Discussion" --header_information_contextual=2

# Best Model trained with different augmentation data
# Best model setting: current sentence content + Section headers (prepended to preceding, current and trialing sentences) + Contextual information + sentence representation [CLS]
# 1 - (generative by GPT-4) concatenate all augmentation outcomes with the original samples in the training set
python train.py --target="[CLS]" --section_header="both" --bert_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --augmentation_mode=1  --augmentation_file="data/cleaned_generative_sents.csv" --sent_dim=0 --rltv=-1 --section_emb=0 --position_emb=0 --section_dim=0 --train_file="data/all_CONSORT_manual_data.csv" --section_avg_sep="" --test_file="" --mode="contextual" --section="whole" --header_information_contextual=2

# 2 - (generative by GPT-4) split augmentation outcomes into different folders in the training set
python train.py --target="[CLS]" --section_header="both" --bert_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --augmentation_mode=2  --augmentation_file="data/cleaned_generative_sents.csv" --sent_dim=0 --rltv=-1 --section_emb=0 --position_emb=0 --section_dim=0 --train_file="data/all_CONSORT_manual_data.csv" --section_avg_sep="" --test_file="" --mode="contextual" --section="whole" --header_information_contextual=2

# 3 - (rephrasing by GPT-4) add the rephrased data to the training set
python train.py --target="[CLS]" --section_header="both" --bert_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --augmentation_mode=3 --augmentation_file="data/Rewritten_GPT_4.csv" --sent_dim=0 --rltv=-1 --section_emb=0 --position_emb=0 --section_dim=0 --train_file="data/all_CONSORT_manual_data.csv" --section_avg_sep="" --test_file="" --mode="contextual" --section="whole" --header_information_contextual=2

# 4 - (EDA) add the data augmented by EDA method
python train.py --target="[CLS]" --section_header="both" --bert_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --augmentation_mode=4  --augmentation_file="" --sent_dim=0 --rltv=-1 --section_emb=0 --position_emb=0 --section_dim=0 --train_file="data/all_CONSORT_manual_data.csv" --test_file=""  --section_avg_sep="" --mode="contextual" --section="whole" --header_information_contextual=2

# 5 - (UMLS-EDA) add the data augmented by UMLS-EDA method
python train.py --target="[CLS]" --section_header="both" --bert_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --augmentation_mode=5  --augmentation_file="" --sent_dim=0 --rltv=-1 --section_emb=0 --position_emb=0 --section_dim=0 --train_file="data/all_CONSORT_manual_data.csv" --test_file="" --section_avg_sep="" --mode="contextual" --section="whole" --header_information_contextual=2

# Vanilla PubMedBERT for sentence classification setting trained with different augmentation data
# Vanilla PubMedBERT for sentence classification setting: current sentence content
# 1 - (generative by GPT-4) concatenate all augmentation outcomes with the original samples in the training set
python train.py --target="[CLS]" --section_header="none" --bert_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --augmentation_mode=1  --augmentation_file="data/cleaned_generative_sents.csv" --sent_dim=0 --rltv=-1 --section_emb=0 --position_emb=0 --section_dim=0 --train_file="data/all_CONSORT_manual_data.csv" --section_avg_sep="" --test_file="" --mode="single" --section="whole"

# 2 - (generative by GPT-4) split augmentation outcomes into different folders in the training set
python train.py --target="[CLS]" --section_header="none" --bert_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --augmentation_mode=2  --augmentation_file="data/cleaned_generative_sents.csv" --sent_dim=0 --rltv=-1 --section_emb=0 --position_emb=0 --section_dim=0 --train_file="data/all_CONSORT_manual_data.csv" --section_avg_sep="" --test_file="" --mode="single" --section="whole"

# 3 - (rephrasing by GPT-4) add the rephrased data to the training set
python train.py --target="[CLS]" --section_header="none" --bert_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --augmentation_mode=3 --augmentation_file="data/Rewritten_GPT_4.csv" --sent_dim=0 --rltv=-1 --section_emb=0 --position_emb=0 --section_dim=0 --train_file="data/all_CONSORT_manual_data.csv" --section_avg_sep="" --test_file="" --mode="single" --section="whole"

# 4 - (EDA) add the data augmented by EDA method
python train.py --target="[CLS]" --section_header="none" --bert_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --augmentation_mode=4  --augmentation_file="" --sent_dim=0 --rltv=-1 --section_emb=0 --position_emb=0 --section_dim=0 --train_file="data/all_CONSORT_manual_data.csv" --test_file="" --mode="single" --section_avg_sep="" --section="whole" --header_information_contextual=2

# 5 - (UMLS-EDA) add the data augmented by UMLS-EDA method
python train.py --target="[CLS]" --section_header="none" --bert_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --augmentation_mode=5  --augmentation_file="" --sent_dim=0 --rltv=-1 --section_emb=0 --position_emb=0 --section_dim=0 --train_file="data/all_CONSORT_manual_data.csv" --test_file="" --mode="single" --section_avg_sep="" --section="whole" --header_information_contextual=2
