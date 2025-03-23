#!/bin/sh
#SBATCH --job-name=YH-V13
#SBATCH -p normal

#SBATCH --qos=a100-8
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:6
#SBATCH --time=60

maindir=$1
# datadir=${maindir}data-100
# codedir=${maindir}code
# taskdir=${datadir}/context-train

datadir=${maindir}data-80
codedir=${maindir}code
taskdir=${datadir}/context-fms-train


RAYGPUS=1
GPU_NUM_PER_NODE=4

# tasks=("MT_DG" "MT_DS" "MT_DHG" "MT_SRE" "MT_DG_SFT" "MT")
# tasks=("MT_DG" "MS_DG" "MT_DS" "MT_DHG" "MT_SRE" "MT_DG_SFT" "MT2" "MS_DS" "MS_DHG" "MS_TIE" "MS_DG_SFT" "MS2")
# tasks=("MS2" "MT2")
tasks=("MS_DG")

# models=("llama-13b")
models=("vicuna-7b-v1.5-16k")
# crosss=("MT_DG" "MT_DS" "MT_DHG" "MT_SRE" "MT_DG_SFT" "MT" "MS_DG" "MS_DS" "MS_DHG" "MS_TIE" "MS_DG_SFT" "MS")
crosss=("MS_DG" "MT_DG")

for model in "${models[@]}"
    do
    raw_model_path=${maindir}pretrained_model/${model}/
    for cross in "${crosss[@]}"
        do
        for task in "${tasks[@]}"
            do
            test_data=${taskdir}/${cross}/test.jsonl
            # zeroshot inference on one node
            # python3 ${codedir}/codes/eval/get_model_infer_simple.py \
            # --model-id ${model}_zeroshot \
            # --model-path ${raw_model_path} \
            # --question-file ${test_data} \
            # --answer-file ${datadir}/instruction_testing/inf_${model}_${task}_zeroshot.jsonl \
            # --num-gpus $GPU_NUM_PER_NODE \
            # --ray-num-gpus ${RAYGPUS}

            model_output_path=${maindir}model/${model}_${task}/
            # # # tuning inference
            python3 ${codedir}/codes/inference/get_model_infer_simple.py \
                --model-id ${model}_${task} \
                --model-path ${model_output_path} \
                --question-file ${test_data} \
                --answer-file ${datadir}/instruction_testing_${cross}/inf_${model}_${task}.jsonl \
                --num-gpus $GPU_NUM_PER_NODE \
                --ray-num-gpus ${RAYGPUS}   
            done
        done
    done