#! /bin/bash

source utils/config.sh

#INPUT_DIR=/home/gyx/nvidia-bert/data/yelp_amazon/yelp_review_full_csv/
INPUT_DIR=/home/gyx/nvidia-bert/data/yelp_amazon/tenk_yelp/
OUTPUT_DIR=/data1/private/zzy/Dynamic-Bert/data/yelp/hdf5_shards
#BERT_MODEL=/home/gyx/nvidia-bert/data/yelp_amazon/yelp_review_full_csv/uncase
BERT_MODEL=/data1/private/zzy/Dynamic-Bert/outputs/yelp_first/uncase_8000
TOP_SEN_RATE=0.8
THRESHOLD=0.2
mkdir -p ${OUTPUT_DIR}
LOWER_CASE_SWITCH=""
if [ "$DO_LOWER_CASE" = true ] ; then
  LOWER_CASE_SWITCH="--do_lower_case" 
fi
# OUTPUT_FILE="${OUTPUT_DIR}/${SHARD_INDEX}.hdf5"
echo "Bert model: ${BERT_MODEL}"

PART=$1

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
  --rand_gen \
  ${LOWER_CASE_SWITCH} 