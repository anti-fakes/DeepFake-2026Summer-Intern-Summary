#!/bin/bash

set -e

# ============================================================
# Evaluation configuration
# ============================================================

num_gpus=2

task=zero_shot
dataset=x_voice_eval

# Config name:
# src/x_voice/configs/XVoice_KO_Replay_SylFix_Stage1.yaml
exp_name=XVoice_Base_Stage1 # XVoice_KO_Replay_SylFix_Stage1

ckpt=600000

# Stage 1 uses reference transcript
drop_text=False

test_set="en en ko"
ref_set="en ko en"

# ============================================================
# Inference settings
# ============================================================

seed=0
nfe=16
speed=1.0

# Stage 1 uses reference transcript
sp_type=utf

srp_exp_name=SpeedPredict_Multilingual
srp_ckpt=28000

# CFG settings
cfg_schedule=square
cfg_decay_time=0.6

# Acoustic CFG
cfg_strength=2.5

# Linguistic CFG
cfg_strength2=4.0

decoupled=True
reverse=False


# ============================================================
# Prepare arguments
# ============================================================

if [ -z "${ref_set}" ]; then
    ref_set="${test_set}"
fi

test_langs=($test_set)
ref_langs=($ref_set)

if [ ${#test_langs[@]} -ne ${#ref_langs[@]} ]; then
    echo "[ERROR] test_set and ref_set must have the same number of languages for cross-lingual eval."
    exit 1
fi


drop_text_args=""

if [ "$drop_text" = "True" ]; then
    drop_text_args="--drop_text"
fi


layered_args=""

if [ "$decoupled" = "True" ]; then
    layered_args="--layered"
fi


reverse_args=""

if [ "$reverse" = "True" ]; then
    reverse_args="--reverse"
fi


# ============================================================
# Project paths
# ============================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../../" && pwd )"

cv3_dir="${PROJECT_ROOT}/data/${dataset}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"


# ============================================================
# Decode directory
# ============================================================

if [ "${decoupled}" = "True" ]; then

    decode_dir="${PROJECT_ROOT}/results/${exp_name}_${ckpt}/${dataset}/${sp_type}_seed${seed}_euler_nfe${nfe}_vocos_ss-1_schedule${cfg_schedule}_cfgI${cfg_strength}_cfgII${cfg_strength2}_speed${speed}zero_shot"

else

    decode_dir="${PROJECT_ROOT}/results/${exp_name}_${ckpt}/${dataset}/${sp_type}_seed${seed}_euler_nfe${nfe}_vocos_ss-1_schedule${cfg_schedule}_cfg${cfg_strength}_speed${speed}zero_shot"

fi


echo
echo "=================================================="
echo " X-Voice Evaluation"
echo "=================================================="
echo " Experiment : ${exp_name}"
echo " Checkpoint : ${ckpt}"
echo " Test set   : ${test_set}"
echo " Ref set    : ${ref_set}"
echo " GPUs       : ${num_gpus}"
echo " Decode dir : ${decode_dir}"
echo "=================================================="
echo


# ============================================================
# Inference
# ============================================================

accelerate launch \
    --main_process_port 29507 \
    --num_processes ${num_gpus} \
    "${PROJECT_ROOT}/src/x_voice/eval/eval_infer_batch_original.py" \
    -s ${seed} \
    -n "${exp_name}" \
    -c ${ckpt} \
    -t "${dataset}" \
    -nfe ${nfe} \
    --speed ${speed} \
    -l "${test_set// /,}" \
    -rl "${ref_set// /,}" \
    --cfg_strength ${cfg_strength} \
    ${layered_args} \
    --cfg_strength2 ${cfg_strength2} \
    --cfg_schedule "${cfg_schedule}" \
    --cfg_decay_time ${cfg_decay_time} \
    --normalize_text \
    --post_processing \
    --decode_dir "${decode_dir}" \
    --sp_type ${sp_type} \
    -ns "${srp_exp_name}" \
    -cs ${srp_ckpt} \
    ${drop_text_args} \
    ${reverse_args}


# ============================================================
# Evaluation
# ============================================================

cd "$SCRIPT_DIR"

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"


DNSMOS_LAB=utils/DNSMOS

. utils/parse_options.sh || exit 1


dumpdir=${cv3_dir}/${task}

test_gt="false"


for ((i=0; i<${#test_langs[@]}; i++)); do

    lang=${test_langs[$i]}
    ref_lang=${ref_langs[$i]}


    # ========================================================
    # WER
    # ========================================================

    echo
    echo "[INFO] Scoring WER for ${decode_dir}/${ref_lang}_${lang}"
    echo

    bash utils/cal_wer.sh \
        ${dumpdir}/${lang}/text \
        ${decode_dir}/${ref_lang}_${lang} \
        ${lang} \
        ${num_gpus} \
        "${test_gt}"


    # ========================================================
    # wav.scp for SIM
    # ========================================================

    find ${decode_dir}/${ref_lang}_${lang}/wavs \
        -name "*.wav" \
        | awk -F '/' '{print $NF, $0}' \
        | sed "s@\.wav @ @g" \
        > ${decode_dir}/${ref_lang}_${lang}/wav.scp


    # ========================================================
    # SIM
    # ========================================================

    echo
    echo "[INFO] Scoring SIM: Comparing ${lang} output with ${ref_lang} prompt"
    echo

    PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}" \
    /data/minseo/XVoice_eval/sim_env/bin/python \
        eval_similarity.py \
        --wavlm_ckpt_dir "${PROJECT_ROOT}/wavlm_large_finetune.pth" \
        --prompt_wavs "${dumpdir}/${ref_lang}/prompt_wav.scp" \
        --hyp_wavs "${decode_dir}/${ref_lang}_${lang}/wav.scp" \
        --log_file "${decode_dir}/${ref_lang}_${lang}/spk_simi_scores.txt" \
        --num_gpus ${num_gpus} \
        --decode_dir "${decode_dir}" \
        --dump_dir "${cv3_dir}"


    # ========================================================
    # UTMOS
    # ========================================================

    echo
    echo "[INFO] Scoring UTMOS for ${decode_dir}/${ref_lang}_${lang}"
    echo

    python eval_utmos.py \
        --audio_dir "${decode_dir}/${ref_lang}_${lang}" \
        --ext "wav"


    # ========================================================
    # DNSMOS
    # Currently disabled
    # ========================================================

    # echo "[INFO] Scoring DNSMOS for ${decode_dir}/${ref_lang}_${lang}"
    #
    # python ${DNSMOS_LAB}/dnsmos_local_wavscp.py \
    #     -t ${decode_dir}/${ref_lang}_${lang}/wav.scp \
    #     -e ${DNSMOS_LAB} \
    #     -o ${decode_dir}/${ref_lang}_${lang}/dnsmos.csv

done


# ============================================================
# Collect final result
# ============================================================

echo
echo "[INFO] Collecting final result"
echo

python collect_results.py \
    --decode_dir "${decode_dir}" \
    --test_set "${test_set}" \
    --ref_set "${ref_set}"


echo
echo "=================================================="
echo " Evaluation finished: checkpoint ${ckpt}"
echo "=================================================="