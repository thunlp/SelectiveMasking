#! /bin/bash

source utils/config.sh

# OUTPUT_FILE="${OUTPUT_DIR}/${SHARD_INDEX}.hdf5"
echo "Bert model: ${BERT_MODEL}"
echo "Input dir: ${INPUT_DIR}"
echo "Output dir: ${OUTPUT_DIR}"

PART=$1
# echo $INPUT_DIR
 CUDA_VISIBLE_DEVICES=$1 python3 ../sc_cpd.py \
  --input_dir=${INPUT_DIR} \
  --output_dir=${OUTPUT_DIR} \
  --max_seq_length=${MAX_SEQUENCE_LENGTH} \
  --max_predictions_per_seq=${MAX_PREDICTIONS_PER_SEQUENCE} \
  --masked_lm_prob=${MASKED_LM_PROB} \
  --random_seed=${SEED} \
  --dupe_factor=${DUPE_FACTOR} \
  --bert_model=${BERT_MODEL} \
  --task_name="yelp" \
  --gpus=${N_PROCS_PREPROCESS} \
  --top_sen_rate=${TOP_SEN_RATE} \
  --part $PART \
  --threshold=${THRESHOLD} \
  --max_proc=${N_PROCS_PREPROCESS} \
  --mode=${MODE} \
  ${LOWER_CASE_SWITCH} 
