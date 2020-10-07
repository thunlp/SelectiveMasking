SEED_ARRAY=(13 43 83 181 271 347 433 659 727 859)

for SEED in ${SEED_ARRAY[@]}
do
    bash scripts/finetune_ckpt.sh ${SEED}
done