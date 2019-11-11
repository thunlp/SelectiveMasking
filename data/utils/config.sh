#! /bin/bash

# set -e

USE_BERT_LARGE=false
MAX_SEQUENCE_LENGTH=256
MAX_PREDICTIONS_PER_SEQUENCE=40
MASKED_LM_PROB=0.15
SEED=12345
DUPE_FACTOR=1
DO_LOWER_CASE=true
DO_WITH_RAND=true
N_LINES_PER_SHARD_APPROX=99000   # Default=396000 creates 256 shards

N_PROCS_PREPROCESS=4    # Adjust this based on memory requirements and available number of cores

BERT_BASE_DIR_CASED="/home/gyx/nvidia-bert/vocab/cased_L-12_H-768_A-12"
BERT_LARGE_DIR_CASED="/home/gyx/nvidia-bert/vocab/cased_L-24_H-1024_A-16"
BERT_BASE_DIR_UNCASED="/home/gyx/nvidia-bert/vocab/uncased_L-12_H-768_A-12"
BERT_LARGE_DIR_UNCASED="/home/gyx/nvidia-bert/vocab/uncased_L-24_H-1024_A-16"

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

MODE=model
TASK_NAME=amazon

# INPUT_DIR=/home/gyx/nvidia-bert/data/yelp_amazon/yelp_review_full_csv/
# INPUT_DIR=/home/gyx/nvidia-bert/data/yelp_amazon/tenk_yelp/

# INPUT_DIR=/home/gyx/nvidia-bert/data/yelp_amazon/tenk_amazon/
# INPUT_DIR=/home/gyx/nvidia-bert/data/yelp_amazon/onem_amazon/
# INPUT_DIR=/home/gyx/nvidia-bert/data/yelp_amazon/amazon_review_full_csv/tmp1/
# INPUT_DIR=/home/gyx/nvidia-bert/data/yelp_amazon/amazon_review_full_csv/tmp2/
INPUT_DIR=/home/gyx/nvidia-bert/data/yelp_amazon/amazon_review_full_csv/tmp3/
# INPUT_DIR=/home/gyx/nvidia-bert/data/rt-polaritydata/full
# INPUT_DIR=/home/gyx/nvidia-bert/data/rt-polaritydata/10k
# INPUT_DIR=/home/gyx/nvidia-bert/data/twitter/small
# INPUT_DIR=/home/gyx/nvidia-bert/data/twitter/full

# OUTPUT_DIR=/home/gyx/nvidia-bert/data/mr_mask/hdf5_shards
# OUTPUT_DIR=/home/gyx/nvidia-bert/data/mr_mask_no_stop/hdf5_shards
# OUTPUT_DIR=/home/gyx/nvidia-bert/data/amazon_mr_model_gen_rand_3/hdf5_shards
# OUTPUT_DIR=/home/gyx/nvidia-bert/data/mr_mask_no_stop/amazon_1/hdf5_shards
# OUTPUT_DIR=/home/gyx/nvidia-bert/data/mr_mask_no_stop/yelp_all/hdf5_shards

# OUTPUT_DIR=/home/gyx/nvidia-bert/data/twitter/test/
# OUTPUT_DIR=/home/gyx/nvidia-bert/data/twitter/twitter_mask/
# OUTPUT_DIR=/home/gyx/nvidia-bert/data/twitter/amazon_test/
OUTPUT_DIR=/home/gyx/nvidia-bert/data/twitter/amazon_3/




# model to generate mask trainning sets
# BERT_MODEL=/home/gyx/nvidia-bert/data/yelp_amazon/yelp_review_full_csv/uncase
# BERT_MODEL=/home/gyx/nvidia-bert/outputs/yelp_first/uncase_8000/
# BERT_MODEL=/home/gyx/nvidia-bert/results/yelp_all_rand_10dup/finetune/ckpt_60000/
# BERT_MODEL=/home/gyx/nvidia-bert/results/yelp_all_rand_10dup/finetune/temp/
# BERT_MODEL=../outputs/yelp_second/uncase_25000/
# BERT_MODEL=../outputs/yelp_full
# BERT_MODEL=/home/gyx/nvidia-bert/results/mr/temp/
# BERT_MODEL=/home/gyx/nvidia-bert/results/yelp_all_rand_10dup/mask_gen/

# mask generator
# BERT_MODEL=/home/gyx/nvidia-bert/results/yelp_all_rand_10dup/mask_generator/temp
# BERT_MODEL=/home/gyx/nvidia-bert/results/mr_t/mask_generator/temp
# BERT_MODEL=/home/gyx/nvidia-bert/results/mr_mask_no_stop/mask_generator/temp
# BERT_MODEL=/home/gyx/nvidia-bert/results/twitter_test/tmp/
BERT_MODEL=/home/gyx/nvidia-bert/results/twitter/mask_generator/tmp/

# OUTPUT_DIR=/home/gyx/nvidia-bert/data/yelp_model_mask/hdf5_shards
# OUTPUT_DIR=/home/gyx/nvidia-bert/data/yelp_mask_info/hdf5_shards

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