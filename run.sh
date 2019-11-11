# DATA_DIR=/home/gyx/nvidia-bert/data/yelp_amazon/yelp_review_full_csv
    # --bert_model bert-base-uncased \
# DATA_DIR=/home/gyx/nvidia-bert/data/rt-polaritydata/full/
DATA_DIR=/home/gyx/nvidia-bert/data/twitter/full/


ITER=$1
SEED=$2

# OUTPUT_DIR=results/twitter/model_amazon/${SEED}/ckpt_${ITER}
OUTPUT_DIR=results/twitter/origin

python3 run_classifier_ckpy.py \
    --bert_model=pretrain_bert_model/bert-base-uncased/ \
    --do_train \
    --do_eval \
    --task_name=twitter \
    --data_dir=${DATA_DIR}/ \
    --output_dir=${OUTPUT_DIR}  \
    --max_seq_length=256   \
    --train_batch_size=32 \
    --num_train_epochs=1 \
    --learning_rate=2e-5 \
    --do_lower_case \
    --fp16 \
    --gradient_accumulation_steps 2 \
    --seed=${SEED} \
    # --ckpt=results/twitter/model_amazon/checkpoints/ckpt_${ITER}.pt \
