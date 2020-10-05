DATA_DIR=${HOME}/SelectiveMasking/data2/absa_lap/30w_absa_mask/merged

OUTPUT_DIR=${HOME}/SelectiveMasking/results2/absa_lap/30w_mask_generator/

BERT_MODEL=${HOME}/SelectiveMasking/pretrain_bert_model/bert-base-uncased/

python3 ${HOME}/SelectiveMasking/mask_model_pretrain.py \
    --bert_model=${BERT_MODEL} \
    --do_eval \
    --task_name=MaskGen \
    --data_dir=${DATA_DIR}/  \
    --output_dir=${OUTPUT_DIR}  \
    --max_seq_length=128   \
    --train_batch_size=32 \
    --num_train_epochs=10 \
    --learning_rate=1e-5 \
    --do_lower_case \
    --gradient_accumulation_steps 2 \
    --rate=3 \
    --ckpt=${CKPT} \
    --do_train \
    # --fp16 \