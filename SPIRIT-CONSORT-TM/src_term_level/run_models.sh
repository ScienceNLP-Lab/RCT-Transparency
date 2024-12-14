#!/bin/bash

ROOT=$PWD

# Assign directory for your own virtual environment, and also the venv name.
# e.g. source /jet/home/ghong1/miniconda3/bin/activate CONSORT-TERM
source {venv-dir} {venv-name}
echo "Activated virtual environment!!"

task=consort
data_dir=../data/terms/processed_data/

# Assign your own output directory
# We'd recommend to assign your own output path to large storage system
output_dir=./output/

# NER Hyperparameters (set your own values if needed)
n_epochs=200
ner_plm_lr=1e-5
ner_task_lr=5e-4
ner_cw=0
max_seq_length=200
max_span_len_ent=10
ner_patience=5
header_type=outermost
SEED=0

#### TASK: SpanNER with PURE model ####

pipeline_task=entity
MODEL=microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext

# Activate if you want to re-use the created pytorch model that you fine-tuned
# entity_output_dir=${output_dir}/EXP_{}/entity

# You can replicate the training process using the script below
python run_term_extraction.py \
    --task $task --pipeline_task $pipeline_task \
    --do_train --do_eval --do_predict_dev \
    --output_dir $output_dir \
    --data_dir "${data_dir}${dataset}" \
    --context_window $ner_cw --max_seq_length $max_seq_length \
    --train_batch_size 32  --eval_batch_size 32 \
    --learning_rate $ner_plm_lr --task_learning_rate $ner_task_lr \
    --num_epoch $n_epochs --eval_per_epoch 1.0 --max_patience $ner_patience \
    --model $MODEL \
    --max_span_length_entity $max_span_len_ent \
    --seed $SEED \
    --use_section_headers $header_type \
    --add_relative_position --num_segment_doc 30 \
    # --negative_sampling_test 1.0 \
    # --entity_output_dir $entity_output_dir \    
    # --load_saved_model \
