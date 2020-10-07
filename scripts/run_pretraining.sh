#!/bin/bash

PER_GPU_TRN_BS=28
LR=2e-5
NUM_GPUS=4
WARM_UP=0.2
TRN_STEPS=50
SAVE_STEPS=4
GRAD_ACC=1
SEED=88

DATA_DIR=data/datasets/test/full_amazon/model/merged/
BERT_MODEL=${HOME}/SelectiveMasking/pretrain_bert_model/bert-base-uncased/
RESULTS_DIR=${HOME}/SelectiveMasking/results/test/full_amazon/model/
CHECKPOINTS_DIR=${RESULTS_DIR}/checkpoints/

mkdir -p $CHECKPOINTS_DIR

echo $DATA_DIR
INPUT_DIR=$DATA_DIR
CMD="run_pretraining.py"
CMD+=" --input_dir=$DATA_DIR"
CMD+=" --output_dir=$CHECKPOINTS_DIR"
CMD+=" --ckpt=${CKPT}"
CMD+=" --bert_model=${BERT_MODEL}"
CMD+=" --train_batch_size=${PER_GPU_TRN_BS}"
CMD+=" --max_seq_length=256"
CMD+=" --max_predictions_per_seq=80"
CMD+=" --max_steps=$TRN_STEPS"
CMD+=" --warmup_proportion=$WARM_UP"
CMD+=" --num_steps_per_checkpoint=$SAVE_STEPS"
CMD+=" --learning_rate=$LR"
CMD+=" --seed=$SEED"
CMD+=" --fp16"

if [ "$NUM_GPUS" -gt 1  ] ; then
   CMD="python3 -m torch.distributed.launch --nproc_per_node=$NUM_GPUS $CMD"
else
   CMD="python3  $CMD"
fi

echo ${CMD}

${CMD}