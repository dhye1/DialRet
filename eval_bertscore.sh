#!/bin/sh
maindir=$1
datadir=${maindir}data-80
codedir=${maindir}code

infdir=${datadir}/instruction_testing_v2_13b_is_1e
refdir=${datadir}/context-fms
outdir=${datadir}/eval_result_bert_v2_13b_is_1e
path_suffix=""

models=("llama-7b")
# model_tuned="MT_DG,MT_DS,MT_DHG,MT_SRE,MT_DG_SFT,MT,MS_DG,MS_DS,MS_DHG,MS_TIE,MS_DG_SFT,MS"
# model_tuned="MT_DG,MT,MT2,MS_DG,MS,MS2"
# model_tuned="MT_DG,MT2,MS_DG,MS2"
# cross_domains=("MT_DG" "MS_DG" "MT" "MS" "indomain" )
# cross_domains=("indomain" "MT_DG" "MS_DG")
# cross_domains=("MT_DG" "MS_DG" "MT_DG_SFT" "MS_DG_SFT")
# cross_domains=("MT_DG_SFT" "MS_DG_SFT")

# model_tuned="MT_DG,MT_DG_SFT,MS_DG_SFT,MS_DG"
model_tuned="MS_DG"
#,MT_DS,MT_DHG,MT_SRE,MT,MS_DS,MS_DHG,MS_TIE,MS"
cross_domains=("MT_DG" "MS_DG")

cpu_thread_num=96

## Indomain trained model
for model in "${models[@]}"
    do
    for cross_domain in "${cross_domains[@]}"
        do
        inf_path=${infdir}${path_suffix}
        ref_path=${refdir}${path_suffix}

        python3 ${codedir}/codes/eval/eval_model_bertscore.py \
            --model-id ${model} \
            --model-tuned ${model_tuned} \
            --cross-domain ${cross_domain} \
            --inference-path ${inf_path} \
            --reference-path ${ref_path} \
            --output-path ${outdir} \
            --cpu-thread-num ${cpu_thread_num}
        done
    done