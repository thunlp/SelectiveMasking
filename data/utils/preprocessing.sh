#! /bin/bash

SHARD_INDEX=${1}
CUDA_ID=${SHARD_INDEX}
# INPUT_FILE="${TARGET_DIR}/final_text_files_sharded/corpus.segmented.part.${SHARD_INDEX}.txt"
INPUT_PREFIX="corpus.segmented.part."
INPUT_DIR="${TARGET_DIR}/final_text_files_sharded/"

source $HOME/nvidia-bert/data/utils/config.sh

OUTPUT_DIR=${TARGET_DIR}/hdf5_shards
mkdir -p ${OUTPUT_DIR}

# OUTPUT_FILE="${OUTPUT_DIR}/${SHARD_INDEX}.hdf5"
echo "Bert model: ${BERT_MODEL}"
CUDA_VISIBLE_DEVICES=${SHARD_INDEX} python3 $HOME/nvidia-bert/create_pretraining_data.py \
  --input_dir=${INPUT_DIR} \
  --input_prefix=${INPUT_PREFIX} \
  --output_dir=${OUTPUT_DIR} \
  --vocab_file=${VOCAB_FILE} \
  --do_lower_case \
  --max_seq_length=${MAX_SEQUENCE_LENGTH} \
  --max_predictions_per_seq=${MAX_PREDICTIONS_PER_SEQUENCE} \
  --masked_lm_prob=${MASKED_LM_PROB} \
  --random_seed=${SEED} \
  --dupe_factor=${DUPE_FACTOR} \
  --bert_model=${BERT_MODEL} \
  --task_name="" \
  --downstream_config="${HOME}/nvidia-bert/downstream_config.json" \
  --local_rank=${SHARD_INDEX} \
  --gpus=${N_PROCS_PREPROCESS}

