DATA_DIR=${HOME}/SelectiveMasking/data/datasets/test/full_rule_mask/merged/
OUTPUT_DIR=${HOME}/SelectiveMasking/results/test/full_mask_generator/
BERT_MODEL=${HOME}/SelectiveMasking/pretrain_bert_model/bert-base-uncased/

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

CMD="python3 ${CMD}"

echo ${CMD}

${CMD}