#!/bin/sh
maindir=$1
datadir=${maindir}data
codedir=${maindir}code

infdir=${datadir}/instruction_testing
refdir=${datadir}/context-fms
contextdir=${datadir}/context-fms
outdir=${datadir}/eval_result_api
path_suffix=""

# models=("gpt-3.5-turbo-1106")
# models=("llama-7b")
# model_tuned="MT_DG,MT_DS,MT_DHG,MT_SRE,MT_DG_SFT,MT,MS_DG,MS_DS,MS_DHG,MS_TIE,MS_DG_SFT,MS"
# model_tuned="MS_DS,MS_DHG,MS_TIE"
# model_tuned="MT_DG,MS_DG,MT2,MS2"

# model_tuned="MT_DG",MS_DG,MT,MS"
# cross_domains=("MS_DG")
# cross_domains=("MT_DG" "MS_DG")
# cross_domains=("MT_DG" "MS_DG")


model_tuned="zeroshot"
# models=('openchat/openchat_3.5' 'mistralai/Mistral-7B-v0.1' 'meta-llama/Llama-2-7b-chat-hf' 'lmsys/vicuna-13b-v1.5-16k' )
# models=('meta-llama/Llama-2-13b-chat-hf')
models=('openchat/openchat_3.5')
cross_domains=("MS_DG" "MT_DG")

cpu_thread_num=96

## Indomain trained model
for model in "${models[@]}"
    do
    for cross_domain in "${cross_domains[@]}"
        do
        inf_path=${infdir}${path_suffix}
        ref_path=${refdir}${path_suffix}

        python3 ${codedir}/codes/eval/eval_api_score_json.py \
            --model-id ${model} \
            --model-tuned ${model_tuned} \
            --cross-domain ${cross_domain} \
            --inference-path ${inf_path} \
            --reference-path ${ref_path} \
            --context-path ${contextdir} \
            --output-path ${outdir} \
            --cpu-thread-num ${cpu_thread_num}

        done
    done