# DATA_DIR=/home/gyx/nvidia-bert/data/yelp_mask_test/hdf5_shards/merged
DATA_DIR=/home/gyx/nvidia-bert/data/yelp_mask_info/hdf5_shards/merged
# DATA_DIR=/home/gyx/nvidia-bert/data/small-yelp_mask_info/

BERT_MODEL=/home/gyx/nvidia-bert/pretrain_bert_model/bert-base-uncased/

python3 /home/gyx/nvidia-bert/mask_model_pretrain.py \
    --bert_model=${BERT_MODEL} \
    --do_train \
    --do_eval \
    --task_name=MaskGen \
    --data_dir=${DATA_DIR}/  \
    --output_dir=results/yelp_all_rand_10dup/mask_generator/  \
    --max_seq_length=256   \
    --train_batch_size=64 \
    --num_train_epochs=3 \
    --learning_rate=2e-5 \
    --do_lower_case \
    --fp16 \
    --gradient_accumulation_steps 2 