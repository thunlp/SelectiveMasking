YELP_DIR=/home/gyx/nvidia-bert/data/yelp_amazon/yelp_review_full_csv
    # --bert_model bert-base-uncased \

ITER=$1

python3 run_classifier_ckpy.py \
    --bert_model=pretrain_bert_model/bert-base-uncased/ \
    --do_train \
    --do_eval \
    --task_name=yelp     \
    --data_dir=${YELP_DIR}/  \
    --output_dir=results/yelp_raw/ckpt_${ITER}.pt  \
    --max_seq_length=256   \
    --train_batch_size=32 \
    --num_train_epochs=3 \
    --learning_rate=2e-5 \
    --do_lower_case \
    --ckpt=results/yelp_reverse/checkpoints/ckpt_${ITER}.pt \
    --fp16 \
    --gradient_accumulation_steps 2
