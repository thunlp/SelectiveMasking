source config/bash_config.sh

DATA_DIR=${E_SELECTIVE_MASKING_TRAIN_NN_DATA_DIR}
OUTPUT_DIR=${E_SELECTIVE_MASKING_TRAIN_NN_OUTPUT_DIR}
BERT_MODEL=${E_GENEPT_BERT_MODEL}

CMD="mask_model_pretrain.py"
CMD+=" --bert_model=${BERT_MODEL}"
CMD+=" --task_name=MaskGen"
CMD+=" --data_dir=${DATA_DIR}"
CMD+=" --output_dir=${OUTPUT_DIR}"
CMD+=" --max_seq_length=128  "
CMD+=" --train_batch_size=32"
CMD+=" --num_train_epochs=10"
CMD+=" --learning_rate=1e-5"
CMD+=" --do_lower_case"
CMD+=" --gradient_accumulation_steps 2"
CMD+=" --sample_weight=3"
CMD+=" --do_train"
CMD+=" --do_eval"
# CMD+=" --save_all"

export CUDA_VISIBLE_DEVICES=${E_SELECTIVE_MASKING_TRAIN_NN_GPU_LIST}
CMD="python3 ${CMD}"

echo ${CMD}

${CMD}