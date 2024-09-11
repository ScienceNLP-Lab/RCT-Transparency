#!/bin/bash

# Assign directory for your own virtual environment, and also the venv name.
# e.g. source /jet/home/ghong1/miniconda3/bin/activate CONSORT-TERM
source {venv-dir} {venv-name}
echo "Activated virtual environment!!"

dataset_name=consort

# Designate the experiment number, and the output path for evaluation
IDX=0
output_dir=./output/EXP_${IDX}

task=test
pred_file=entity/ent_pred_$task.json

python run_eval.py \
    --prediction_file "${output_dir}/${pred_file}" \
    --output_dir ${output_dir} \
    --task $task \
    --dataset_name $dataset_name
