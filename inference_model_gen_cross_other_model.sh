#!/bin/sh
#SBATCH --job-name=YH-V13
#SBATCH -p normal

#SBATCH --qos=a100-8
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:6
#SBATCH --time=60

maindir=$1
datadir=${maindir}data
codedir=${maindir}code
taskdir=${datadir}/tasks

RAYGPUS=1
GPU_NUM_PER_NODE=4

# tasks=("MT_DG" "MT_DS" "MT_DHG" "MT_SRE" "MT_DG_SFT" "MT")
# tasks=("MT_DG" "MT_DS" "MT_DHG" "MT_SRE" "MT_DG_SFT" "MT" "MS_DG" "MS_DS" "MS_DHG" "MS_TIE" "MS_DG_SFT" "MS")
tasks=("MS2")
models=("vicuna-7b-v1.5-16k" "Mistral-7B-v0.1" "Llama-2-7b-chat-hf" "openchat_3.5")
# crosss=("MT_DG" "MT_DS" "MT_DHG" "MT_SRE" "MT_DG_SFT" "MT" "MS_DG" "MS_DS" "MS_DHG" "MS_TIE" "MS_DG_SFT" "MS")
crosss=("MT_DG" "MS_DG")

for model in "${models[@]}"
    do
    raw_model_path=${maindir}pretrained_model/${model}/
    for cross in "${crosss[@]}"
        do
        for task in "${tasks[@]}"
            do
            test_data=${taskdir}/${cross}/test.jsonl
            # zeroshot inference on one node
            python3 ${codedir}/codes/inference/get_model_infer_simple.py \
            --model-id ${model}_zeroshot \
            --model-path ${raw_model_path} \
            --question-file ${test_data} \
            --answer-file ${datadir}/instruction_testing/inf_${model}_${task}_zeroshot.jsonl \
            --num-gpus $GPU_NUM_PER_NODE \
            --ray-num-gpus ${RAYGPUS}

            # model_output_path=${maindir}model/${model}_${task}/
            # # # # tuning inference
            # python3 ${codedir}/codes/inference/get_model_infer_simple.py \
            #     --model-id ${model}_${task} \
            #     --model-path ${model_output_path} \
            #     --question-file ${test_data} \
            #     --answer-file ${datadir}/instruction_testing_other_${cross}/inf_${model}_${task}.jsonl \
            #     --num-gpus $GPU_NUM_PER_NODE \
            #     --ray-num-gpus ${RAYGPUS}   
            done
        done
    done