source data/create_data_rule/config.sh

mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}/merged/

source data/create_data_rule/xarg_wrapper.sh

python3 data/merge_mask_data.py ${OUTPUT_DIR} ${MAX_PROC}

