# DATA_DIR=${HOME}/nvidia-bert/data/yelp_mask_test/hdf5_shards/merged
# DATA_DIR=${HOME}/nvidia-bert/data/yelp_mask_info/hdf5_shards/merged
# DATA_DIR=${HOME}/nvidia-bert/data/small-yelp_mask_info/
# DATA_DIR=${HOME}/nvidia-bert/data/mr_mask_no_stop/hdf5_shards/merged
# DATA_DIR=${HOME}/nvidia-bert/data/twitter/twitter_mask/merged
DATA_DIR=${HOME}/nvidia-bert/data/absa/30w_absa_mask/merged

# OUTPUT_DIR=${HOME}/nvidia-bert/results/mr_mask_no_stop/mask_generator/
# OUTPUT_DIR=${HOME}/nvidia-bert/results/twitter/mask_generator/
OUTPUT_DIR=${HOME}/nvidia-bert/results/absa/30w_mask_generator/

BERT_MODEL=${HOME}/nvidia-bert/pretrain_bert_model/bert-base-uncased/
CKPT=${HOME}/nvidia-bert/results/small_bert/ckpt_300000.pt

python3 ${HOME}/nvidia-bert/mask_model_pretrain.py \
    --bert_model=${BERT_MODEL} \
    --do_train \
    --do_eval \
    --task_name=MaskGen \
    --data_dir=${DATA_DIR}/  \
    --output_dir=${OUTPUT_DIR}  \
    --max_seq_length=128   \
    --train_batch_size=32 \
    --num_train_epochs=10 \
    --learning_rate=1e-5 \
    --do_lower_case \
    --fp16 \
    --gradient_accumulation_steps 2 \
    --ckpt=${CKPT}