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

MODE=model
TASK_NAME=yelp

# INPUT_DIR=${HOME}/nvidia-bert/data/yelp_amazon/yelp_review_full_csv/
# INPUT_DIR=${HOME}/nvidia-bert/data/yelp_amazon/tenk_yelp/

# INPUT_DIR=${HOME}/nvidia-bert/data/yelp_amazon/tenk_amazon/
# INPUT_DIR=${HOME}/nvidia-bert/data/yelp_amazon/onem_amazon/
INPUT_DIR=${HOME}/nvidia-bert/data/yelp_amazon/amazon_review_full_csv/tmp1/
# INPUT_DIR=${HOME}/nvidia-bert/data/yelp_amazon/amazon_review_full_csv/tmp2/
# INPUT_DIR=${HOME}/nvidia-bert/data/yelp_amazon/amazon_review_full_csv/tmp3/


# OUTPUT_DIR=${HOME}/nvidia-bert/data/mr_mask_no_stop/amazon_1/hdf5_shards
# OUTPUT_DIR=${HOME}/nvidia-bert/data/mr_mask_no_stop/yelp_all/hdf5_shards
# OUTPUT_DIR=${HOME}/nvidia-bert/data/mr_mask_no_stop/30w_mr_mask/
# OUTPUT_DIR=${HOME}/nvidia-bert/data/mr_mask_no_stop/30w_model_amazon_3/



# OUTPUT_DIR=${HOME}/nvidia-bert/data/absa/absa_mask
# OUTPUT_DIR=${HOME}/nvidia-bert/data/absa/amazon_3
# OUTPUT_DIR=${HOME}/nvidia-bert/data/absa/30w_absa_mask/
# OUTPUT_DIR=${HOME}/nvidia-bert/data/absa/30w_model_yelp/
# OUTPUT_DIR=${HOME}/nvidia-bert/data/absa/30w_model_amazon_2/
# OUTPUT_DIR=${HOME}/nvidia-bert/data/absa/20w_absa_mask/
# OUTPUT_DIR=${HOME}/nvidia-bert/data/absa/20w_model_yelp/
# OUTPUT_DIR=${HOME}/nvidia-bert/data/absa/20w_model_amazon_1/
# OUTPUT_DIR=${HOME}/nvidia-bert/data/absa/20w_model_amazon_2/
# OUTPUT_DIR=${HOME}/nvidia-bert/data/absa/20w_model_amazon_3/

# OUTPUT_DIR=${HOME}/nvidia-bert/data/absa/10w_model_amazon_3/
# OUTPUT_DIR=${HOME}/nvidia-bert/data/absa/10w_model_amazon_2/
# OUTPUT_DIR=${HOME}/nvidia-bert/data/absa/10w_model_amazon_1/

# OUTPUT_DIR=${HOME}/nvidia-bert/data/mr_mask_no_stop/10w_model_yelp/
# OUTPUT_DIR=${HOME}/nvidia-bert/data/mr_mask_no_stop/10w_model_amazon_1/
# OUTPUT_DIR=${HOME}/nvidia-bert/data/mr_mask_no_stop/10w_model_amazon_2/
# OUTPUT_DIR=${HOME}/nvidia-bert/data/mr_mask_no_stop/10w_model_amazon_3/

# OUTPUT_DIR=${HOME}/nvidia-bert/data/mr_mask_no_stop/20w_model_amazon_1/
# OUTPUT_DIR=${HOME}/nvidia-bert/data/mr_mask_no_stop/20w_model_amazon_2/
# OUTPUT_DIR=${HOME}/nvidia-bert/data/mr_mask_no_stop/10w_model_amazon_3/

# OUTPUT_DIR=${HOME}/nvidia-bert/data/absa_lap/full_model_yelp/
# OUTPUT_DIR=${HOME}/nvidia-bert/data/absa_lap/full_model_amazon/tmp3
# OUTPUT_DIR=${HOME}/nvidia-bert/data/absa_lap/10w_model_yelp/
# OUTPUT_DIR=${HOME}/nvidia-bert/data2/absa_lap/20w_model_amazon/tmp3
# OUTPUT_DIR=${HOME}/nvidia-bert/data2/absa_lap/20w_model_yelp/
# OUTPUT_DIR=${HOME}/nvidia-bert/data2/absa_lap/30w_model_yelp/
OUTPUT_DIR=${HOME}/nvidia-bert/data2/absa_lap/30w_model_amazon/tmp1


# mask generator
# BERT_MODEL=${HOME}/nvidia-bert/results/yelp_all_rand_10dup/mask_generator/temp
# BERT_MODEL=${HOME}/nvidia-bert/results/mr_t/mask_generator/temp
# BERT_MODEL=${HOME}/nvidia-bert/results/mr_mask_no_stop/mask_generator/temp
# BERT_MODEL=${HOME}/nvidia-bert/results/twitter_test/tmp/
# BERT_MODEL=${HOME}/nvidia-bert/results/twitter/mask_generator/tmp/
# BERT_MODEL=${HOME}/nvidia-bert/results/absa/origin/42/ckpt_0/tmp/
# BERT_MODEL=${HOME}/nvidia-bert/results/absa/small_bert/659/ckpt_300000/tmp/
# BERT_MODEL=${HOME}/nvidia-bert/results/absa/mask_generator/tmp/
# BERT_MODEL=${HOME}/nvidia-bert/results/absa/30w_mask_generator/tmp/
# BERT_MODEL=${HOME}/nvidia-bert/results/mr_mask_no_stop/small_bert/tmp/
# BERT_MODEL=${HOME}/nvidia-bert/results/mr_mask_no_stop/30w_mask_generator/tmp/
# BERT_MODEL=${HOME}/nvidia-bert/results/absa/10w_mask_generator/tmp/
# BERT_MODEL=${HOME}/nvidia-bert/results/absa/20w_mask_generator/tmp/
# BERT_MODEL=${HOME}/nvidia-bert/results/mr_mask_no_stop/10w_mask_generator/tmp/
# BERT_MODEL=${HOME}/nvidia-bert/results/mr_mask_no_stop/10w_mask_generator/tmp/
# BERT_MODEL=${HOME}/nvidia-bert/results/absa_lap/full_mask_generator-2/best
BERT_MODEL=${HOME}/nvidia-bert/results2/absa_lap/30w_mask_generator/best

# OUTPUT_DIR=${HOME}/nvidia-bert/data/yelp_model_mask/hdf5_shards
# OUTPUT_DIR=${HOME}/nvidia-bert/data/yelp_mask_info/hdf5_shards

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