#! /bin/bash
MAX_SEQUENCE_LENGTH=256
MAX_PREDICTIONS_PER_SEQUENCE=40
MASKED_LM_PROB=0.15
SEED=12345
DUPE_FACTOR=1
DO_WITH_RAND=true
N_LINES_PER_SHARD_APPROX=99000

GPU_LIST=(0 1)    # Adjust this based on memory requirements and available number of cores
MAX_PROC=${#GPU_LIST[@]}

MODE=model
TASK_NAME=yelp

INPUT_DIR=${HOME}/SelectiveMasking/data/datasets/yelp_amazon/tenk_yelp

OUTPUT_DIR=${HOME}/SelectiveMasking/data/datasets/test/full_model_yelp/

# model to generate mask training sets
BERT_MODEL=${HOME}/SelectiveMasking/results/test/full_mask_generator/best_model

TOP_SEN_RATE=1
THRESHOLD=0.01

WITH_RAND=""
if [ "$DO_WITH_RAND" = true ] ; then
  WITH_RAND="--with_rand"
fi