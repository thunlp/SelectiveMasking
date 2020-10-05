#! /bin/bash

source data/create_data_model/config.sh

echo "Bert model: ${BERT_MODEL}"
echo "Input dir: ${INPUT_DIR}"
echo "Output dir: ${OUTPUT_DIR}"

GPU_ID=$1
PART=$2

echo "GPU id: ${GPU_ID} PART: ${PART}"
CMD="data/create_data.py"
CMD+=" --input_dir=${INPUT_DIR}"
CMD+=" --output_dir=${OUTPUT_DIR}"
CMD+=" --max_seq_length=${MAX_SEQUENCE_LENGTH}"
CMD+=" --max_predictions_per_seq=${MAX_PREDICTIONS_PER_SEQUENCE}"
CMD+=" --masked_lm_prob=${MASKED_LM_PROB}"
CMD+=" --random_seed=${SEED}"
CMD+=" --dupe_factor=${DUPE_FACTOR}"
CMD+=" --bert_model=${BERT_MODEL}"
CMD+=" --task_name=${TASK_NAME}"
CMD+=" --top_sen_rate=${TOP_SEN_RATE}"
CMD+=" --part ${PART}"
CMD+=" --threshold=${THRESHOLD}"
CMD+=" --max_proc=${MAX_PROC}"
CMD+=" --mode=${MODE}"
CMD+=" --do_lower_case"
CMD+=" ${WITH_RAND}"

export CUDA_VISIBLE_DEVICES=${GPU_ID}
CMD="python3 ${CMD}"

echo ${CMD}

${CMD}