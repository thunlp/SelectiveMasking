SEED=${1:-42}

source config/bash_config.sh

DATA_DIR=${E_SELECTIVE_MASKING_FINETUNE_BERT_DATA_DIR}
BERT_MODEL=${E_GENEPT_BERT_MODEL}
OUTPUT_DIR=${E_SELECTIVE_MASKING_FINETUNE_BERT_OUTPUT_DIR}/${SEED}

CMD="finetune.py"
CMD+=" --bert_model=${BERT_MODEL}"
CMD+=" --do_train"
CMD+=" --do_eval"
CMD+=" --task_name=absa_term"
CMD+=" --data_dir=${DATA_DIR}"
CMD+=" --output_dir=${OUTPUT_DIR} "
CMD+=" --max_seq_length=256  "
CMD+=" --train_batch_size=32"
CMD+=" --num_train_epochs=10"
CMD+=" --learning_rate=2e-5"
CMD+=" --do_lower_case"
CMD+=" --gradient_accumulation_steps 2"
CMD+=" --seed=${SEED}"
CMD+=" --fp16"

export CUDA_VISIBLE_DEVICES=${E_SELECTIVE_MASKING_FINETUNE_BERT_GPU_LIST}

CMD="python3 ${CMD}"

echo ${CMD}

${CMD}