#! /bin/bash
MAX_SEQUENCE_LENGTH=128
MAX_PREDICTIONS_PER_SEQUENCE=20
MASKED_LM_PROB=0.15
SEED=12345
DUPE_FACTOR=1
DO_WITH_RAND=false
N_LINES_PER_SHARD_APPROX=99000

GPU_LIST=(0 1)    # Adjust this based on memory requirements and available number of cores
MAX_PROC=${#GPU_LIST[@]}

MODE=rule
TASK_NAME=absa_term

INPUT_DIR=${HOME}/SelectiveMasking/data/datasets/Aspect-Based-Sentiment-Analysis/14data_lap/

OUTPUT_DIR=${HOME}/SelectiveMasking/data/datasets/test/full_rule_mask/

# model to generate mask training sets
BERT_MODEL=${HOME}/SelectiveMasking/results/test/origin/42/ckpt_1M/best_model/

TOP_SEN_RATE=1
THRESHOLD=0.01

WITH_RAND=""
if [ "$DO_WITH_RAND" = true ] ; then
  WITH_RAND="--with_rand"
fi