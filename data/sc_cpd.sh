#! /bin/bash

source $HOME/nvidia-bert/data/utils/config.sh

INPUT_DIR=$HOME/nvidia-bert/data/yelp_amazon/tenk_yelp/
OUTPUT_DIR=$HOME/nvidia-bert/data/yelp/hdf5_shards
BERT_MODEL=$HOME/nvidia-bert/yelp_out/
mkdir -p ${OUTPUT_DIR}
LOWER_CASE_SWITCH=""
if [ "$DO_LOWER_CASE" = true ] ; then
  LOWER_CASE_SWITCH="--do_lower_case" 
fi
# OUTPUT_FILE="${OUTPUT_DIR}/${SHARD_INDEX}.hdf5"
echo "Bert model: ${BERT_MODEL}"
python3 $HOME/nvidia-bert/sc_cpd.py \
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
  ${LOWER_CASE_SWITCH} 