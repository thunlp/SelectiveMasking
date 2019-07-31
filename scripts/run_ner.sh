#!/usr/bin/env bash

init_checkpoint=${1:-"$HOME/checkpoints/bert_uncased.pt"}
epochs=${2:-"2.0"}
batch_size=${3:-"3"}
learning_rate=${4:-"3e-5"}
precision=${5:-"fp16"}
num_gpu=${6:-"8"}
seed=${7:-"1"}
conll_dir=${8:-"/home/gyx/nvidia-bert/data/CoNll"}
vocab_file=${9:-"/home/gyx/nvidia-bert/vocab/vocab"}
OUT_DIR=${10:-"$HOME/nvidia-bert/results/CoNll"}
mode=${11:-"train eval"}
CONFIG_FILE=${12:-"$HOME/nvidia-bert/bert_config/bert_base_config.json"}
max_steps=${13:-"-1"}
max_seq_length=${14:-"512"}

echo "out dir is $OUT_DIR"
# mkdir -p $OUT_DIR
# if [ ! -d "$OUT_DIR" ]; then
#   echo "ERROR: non existing $OUT_DIR"
#   exit 1
# fi

use_fp16=""
if [ "$precision" = "fp16" ] ; then
  echo "fp16 activated!"
  use_fp16=" --fp16 "
fi

if [ "$num_gpu" = "1" ] ; then
  export CUDA_VISIBLE_DEVICES=0
  mpi_command=""
else
  unset CUDA_VISIBLE_DEVICES
  mpi_command=" -m torch.distributed.launch --nproc_per_node=$num_gpu"
fi

CMD="python3  $mpi_command run_ner.py "
CMD+="--init_checkpoint=$init_checkpoint "
CMD+="--task_name=ner "
if [ "$mode" = "train" ] ; then
  CMD+="--do_train "
  CMD+="--data_dir=$conll_dir "
  CMD+="--train_batch_size=$batch_size "
elif [ "$mode" = "eval" ] ; then
  CMD+="--do_eval "
  CMD+="--data_dir=$conll_dir "
  CMD+="--eval_batch_size=$batch_size "
else
  CMD+="--do_train "
  CMD+="--data_dir=$conll_dir "
  CMD+="--train_batch_size=$batch_size "
  CMD+="--do_eval "
  CMD+="--eval_batch_size=$batch_size "
fi
CMD+=" --do_lower_case "
# CMD+=" --old "
# CMD+=" --loss_scale=128 "
CMD+=" --bert_model=bert-base-uncased "
CMD+=" --learning_rate=$learning_rate "
CMD+=" --seed=$seed "
CMD+=" --num_train_epochs=$epochs "
CMD+=" --max_seq_length=$max_seq_length "
CMD+=" --output_dir=$OUT_DIR "
CMD+=" --vocab_file=$vocab_file "
CMD+=" --config_file=$CONFIG_FILE "
CMD+=" --max_steps=$max_steps "
CMD+=" $use_fp16"

LOGFILE=$OUT_DIR/logfile.txt
echo "$CMD |& tee $LOGFILE"
time $CMD |& tee $LOGFILE

#sed -r 's/
#|([A)/\n/g' $LOGFILE > $LOGFILE.edit

if [ "$mode" != "eval" ]; then
throughput=`cat $LOGFILE | grep -E 'Iteration.*[0-9.]+(it/s)' | tail -1 | egrep -o '[0-9.]+(s/it|it/s)' | head -1 | egrep -o '[0-9.]+'`
train_perf=$(awk 'BEGIN {print ('$throughput' * '$num_gpu' * '$batch_size')}')
echo " training throughput: $train_perf"
fi
