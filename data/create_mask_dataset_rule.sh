source utils/config_rule.sh

mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}/merged/

source xarg_sc_cpd_rule.sh


python3 ${HOME}/nvidia-bert/merge_mask_data.py ${OUTPUT_DIR} ${N_PROCS_PREPROCESS}

