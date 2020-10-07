#!/bin/bash
source config/bash_config.sh

PER_GPU_TRN_BS=28
LR=2e-5
NUM_GPUS=4
WARM_UP=0.2
TRN_STEPS=50
SAVE_STEPS=4
GRAD_ACC=1
SEED=88

DATA_DIR=${E_TASKPT_DATA_DIR}
BERT_MODEL=${E_GENEPT_BERT_MODEL}
RESULTS_DIR=${E_TASKPT_OUTPUT_DIR}
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

export CUDA_VISIBLE_DEVICES=${E_TASKPT_GPU_LIST}
if [ "$NUM_GPUS" -gt 1  ] ; then
   CMD="python3 -m torch.distributed.launch --nproc_per_node=$NUM_GPUS $CMD"
else
   CMD="python3  $CMD"
fi

echo ${CMD}

${CMD}