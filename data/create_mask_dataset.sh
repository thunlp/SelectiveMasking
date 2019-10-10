source utils/config.sh

source xarg_sc_cpd.sh

mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}/merged/

if [ "$MODE" = "rule" ] ; then
    python3 /home/gyx/nvidia-bert/merge_mask_data.py ${OUTPUT_DIR} ${N_PROCS_PREPROCESS}
fi

