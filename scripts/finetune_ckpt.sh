SEED=$1

DATA_DIR=data/datasets/Aspect-Based-Sentiment-Analysis/14data_lap/
BERT_MODEL=pretrain_bert_model/bert-base-uncased/
OUTPUT_DIR=results/test/full_amazon/model/${SEED}
CKPT=results/test/full_amazon/model/checkpoints/best_ckpt.pt

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
CMD+=" --ckpt ${CKPT}"

CMD="python3 ${CMD}"

echo ${CMD}

${CMD}