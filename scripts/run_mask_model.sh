DATA_DIR=/home/gyx/nvidia-bert/data/yelp_mask_test/hdf5_shards/merged
    # --bert_model bert-base-uncased \

BERT_MODEL=/home/gyx/nvidia-bert/pretrain_bert_model/bert-base-uncased/

python3 /home/gyx/nvidia-bert/mask_model_pretrain.py \
    --bert_model=${BERT_MODEL} \
    --do_train \
    --do_eval \
    --task_name=MaskGen \
    --data_dir=${DATA_DIR}/  \
    --output_dir=results/yelp_all_rand_10dup/mask_gen/  \
    --max_seq_length=256   \
    --train_batch_size=32 \
    --num_train_epochs=3 \
    --learning_rate=2e-5 \
    --do_lower_case \
    --gradient_accumulation_steps 2 