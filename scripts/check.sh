SEED_ARRAY=(13 43 83 181 271 347 433 659 727 859)

DIR=$1
ITER=$2

for SEED in ${SEED_ARRAY[@]}
do
    cat ${HOME}/nvidia-bert/${DIR}/${SEED}/ckpt_${ITER}/all_results.txt
    echo ""
done