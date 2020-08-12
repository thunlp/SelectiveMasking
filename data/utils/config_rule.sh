#! /bin/bash

# set -e

USE_BERT_LARGE=false
MAX_SEQUENCE_LENGTH=128
MAX_PREDICTIONS_PER_SEQUENCE=20
MASKED_LM_PROB=0.15
SEED=12345
DUPE_FACTOR=1
DO_LOWER_CASE=true
DO_WITH_RAND=false
N_LINES_PER_SHARD_APPROX=99000   # Default=396000 creates 256 shards

N_PROCS_PREPROCESS=2    # Adjust this based on memory requirements and available number of cores

BERT_BASE_DIR_CASED="${HOME}/nvidia-bert/vocab/cased_L-12_H-768_A-12"
BERT_LARGE_DIR_CASED="${HOME}/nvidia-bert/vocab/cased_L-24_H-1024_A-16"
BERT_BASE_DIR_UNCASED="${HOME}/nvidia-bert/vocab/uncased_L-12_H-768_A-12"
BERT_LARGE_DIR_UNCASED="${HOME}/nvidia-bert/vocab/uncased_L-24_H-1024_A-16"

if [ "$USE_BERT_LARGE" = true ] ; then
  if [ "$DO_LOWER_CASE" = true ] ; then
    VOCAB_FILE="${BERT_LARGE_DIR_UNCASED}/vocab.txt"
    BERT_MODEL="bert-large-uncased"
  else
    VOCAB_FILE="${BERT_LARGE_DIR_CASED}/vocab.txt"
    BERT_MODEL="bert-large-cased"
  fi
else
  if [ "$DO_LOWER_CASE" = true ] ; then
    VOCAB_FILE="${BERT_BASE_DIR_UNCASED}/vocab.txt"
    BERT_MODEL="bert-base-uncased"
  else
    VOCAB_FILE="${BERT_BASE_DIR_CASED}/vocab.txt"
    BERT_MODEL="bert-base-cased"
  fi
fi

MODE=rule
TASK_NAME=absa_term

# INPUT_DIR=${HOME}/nvidia-bert/data/rt-polaritydata/full
# INPUT_DIR=${HOME}/nvidia-bert/data/rt-polaritydata/10k
INPUT_DIR=${HOME}/nvidia-bert/data/Aspect-Based-Sentiment-Analysis/14data_lap/

# OUTPUT_DIR=${HOME}/nvidia-bert/data/absa/20w_absa_mask/
# OUTPUT_DIR=${HOME}/nvidia-bert/data/absa/20w_absa_mask/
OUTPUT_DIR=${HOME}/nvidia-bert/data/absa_lap/full_absa_mask/



# model to generate mask trainning sets
# BERT_MODEL=${HOME}/nvidia-bert/results/mr/temp/
# BERT_MODEL=${HOME}/nvidia-bert/results/yelp_all_rand_10dup/mask_gen/
# BERT_MODEL=${HOME}/nvidia-bert/results/absa/small_bert/tmp_20w/
BERT_MODEL=${HOME}/nvidia-bert/results/absa_lap/origin/42/ckpt_0/

TOP_SEN_RATE=1
THRESHOLD=0.01
LOWER_CASE_SWITCH=""
if [ "$DO_LOWER_CASE" = true ] ; then
  LOWER_CASE_SWITCH="--do_lower_case" 
fi

WITH_RAND=""
if [ "$DO_WITH_RAND" = true ] ; then
  WITH_RAND="--with_rand"
fi