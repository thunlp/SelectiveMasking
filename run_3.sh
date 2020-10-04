# DATA_DIR=${HOME}/nvidia-bert/data/yelp_amazon/yelp_review_full_csv
    # --bert_model bert-base-uncased \
# DATA_DIR=${HOME}/nvidia-bert/data/rt-polaritydata/full/
# DATA_DIR=${HOME}/nvidia-bert/data/twitter/full_pre/
DATA_DIR=${HOME}/nvidia-bert/data/Aspect-Based-Sentiment-Analysis/14data_lap/



ITER=$1
SEED=$2

# OUTPUT_DIR=results/twitter/rand_amazon_pre/${SEED}/ckpt_${ITER}
# OUTPUT_DIR=results/twitter/origin/${SEED}/ckpt_${ITER}
# OUTPUT_DIR=results/twitter/test/${SEED}/ckpt_${ITER}
# OUTPUT_DIR=results/absa/30w_model_amazon/${SEED}/ckpt_${ITER}
# OUTPUT_DIR=results/absa/small_bert/${SEED}/ckpt_${ITER}
# OUTPUT_DIR=results/absa/small_bert/${SEED}/ckpt_${ITER}

# OUTPUT_DIR=results/absa/10w_model_amazon/${SEED}/ckpt_${ITER}

OUTPUT_DIR=results2/absa_lap/30w_model_amazon/${SEED}/ckpt_${ITER}

# OUTPUT_DIR=results/absa_lap/full_model_yelp/${SEED}/ckpt_${ITER}
# OUTPUT_DIR=results/absa_lap/full_rand_yelp/${SEED}/ckpt_${ITER}


# CKPT=results/small_bert/ckpt_${ITER}/pytorch_model.bin
# CKPT=results/absa/20w_rand_yelp/checkpoints/best_ckpt_${ITER}.pt
# CKPT=results/absa_lap/full_rand_yelp/checkpoints/best_ckpt_${ITER}.pt
CKPT=results2/absa_lap/30w_model_amazon/checkpoints/best_ckpt_${ITER}.pt

python3 run_classifier_ckpy.py \
    --bert_model=pretrain_bert_model/bert-base-uncased/ \
    --do_train \
    --do_eval \
    --task_name=absa_term \
    --data_dir=${DATA_DIR}/ \
    --output_dir=${OUTPUT_DIR}  \
    --max_seq_length=256   \
    --train_batch_size=32 \
    --num_train_epochs=10 \
    --learning_rate=2e-5 \
    --do_lower_case \
    --gradient_accumulation_steps 2 \
    --seed=${SEED} \
    --ckpt=${CKPT} \
    --fp16 \
