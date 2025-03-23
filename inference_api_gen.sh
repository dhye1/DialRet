#!/bin/sh
maindir=$1
datadir=${maindir}data
codedir=${maindir}code
taskdir=${datadir}/tasks

# tasks=("MT_DG" "MT_DS" "MT_DHG" "MT_SRE" "MT_DG_SFT" "MT")
tasks=("MS_DS" "MS_DHG" "MS_TIE" "MS_DG_SFT" "MS")
tasks=("MT_DG")
models=("gpt-3.5-turbo-1106")

for model in "${models[@]}"
    do
    for task in "${tasks[@]}"
        do
        test_data=${taskdir}/${task}/test.jsonl

        # # # tuning inference
        python3 ${codedir}/codes/inference/gen_api_answer.py \
            --model-id ${model} \
            --question-file ${test_data} \
            --answer-file ${datadir}/instruction_testing/inf_${model}_${task}.jsonl
        done
    done