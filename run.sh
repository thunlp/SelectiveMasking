# DATA_DIR=/home/gyx/nvidia-bert/data/yelp_amazon/yelp_review_full_csv
    # --bert_model bert-base-uncased \
DATA_DIR=/home/gyx/nvidia-bert/data/rt-polaritydata/full/


ITER=$1

python3 run_classifier_ckpy.py \
    --bert_model=pretrain_bert_model/bert-base-uncased/ \
    --do_train \
    --do_eval \
    --task_name=mr \
    --data_dir=${DATA_DIR}/ \
    --output_dir=results/mr_888/ckpt_${ITER}  \
    --max_seq_length=256   \
    --train_batch_size=32 \
    --num_train_epochs=10 \
    --learning_rate=2e-5 \
    --do_lower_case \
    --fp16 \
    --gradient_accumulation_steps 2 \
    --seed=888 \
    --ckpt=results/mr_model_gen_amazon_1/checkpoints/ckpt_${ITER}.pt \
