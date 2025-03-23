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
# tasks=("MT_DG" "MT_DG_SFT" "MS_DG" "MS_DG_SFT" "MT_DS" "MT_DHG" "MT_SRE"  "MT" "MS_DG" "MS_DS" "MS_DHG" "MS_TIE"  "MS")
tasks=("MT_DG" "MT_DG_SFT" "MS_DG" "MS_DG_SFT")

# models=("t5-small" "t5-base" "t5-large" "t5-3b")
# models=("flan-t5-small" "flan-t5-base" "flan-t5-large" "flan-t5-3b")
# models=("llama-7b" "llama-13b" "llama-33b")
# models=("t5-small" "t5-base" "t5-large" "flan-t5-small" "flan-t5-base" "flan-t5-large"  "t5-3b"  "flan-t5-3b")
models=("llama-13b")

for model in "${models[@]}"
    do
    raw_model_path=${maindir}pretrained_model/${model}/

    for task in "${tasks[@]}"
        do
        test_data=${taskdir}/${task}/test.jsonl
        # test_data=${taskdir}/MT_DG/test.jsonl
        
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
            --answer-file ${datadir}/instruction_testing_indomain/inf_${model}_${task}.jsonl \
            --num-gpus $GPU_NUM_PER_NODE \
            --ray-num-gpus ${RAYGPUS}   
        done
    done