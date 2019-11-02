# DATA_DIR=/home/gyx/nvidia-bert/data/yelp_mask_test/hdf5_shards/merged
# DATA_DIR=/home/gyx/nvidia-bert/data/yelp_mask_info/hdf5_shards/merged
# DATA_DIR=/home/gyx/nvidia-bert/data/small-yelp_mask_info/
DATA_DIR=/home/gyx/nvidia-bert/data/mr_mask_no_stop/hdf5_shards/merged
OUTPUT_DIR=/home/gyx/nvidia-bert/results/mr_mask_no_stop/mask_generator/
BERT_MODEL=/home/gyx/nvidia-bert/pretrain_bert_model/bert-base-uncased/

python3 /home/gyx/nvidia-bert/mask_model_pretrain.py \
    --bert_model=${BERT_MODEL} \
    --do_train \
    --do_eval \
    --task_name=MaskGen \
    --data_dir=${DATA_DIR}/  \
    --output_dir=${OUTPUT_DIR}  \
    --max_seq_length=128   \
    --train_batch_size=32 \
    --num_train_epochs=3 \
    --learning_rate=1e-4 \
    --do_lower_case \
    --fp16 \
    --gradient_accumulation_steps 2 