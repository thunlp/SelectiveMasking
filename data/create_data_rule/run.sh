source data/create_data_rule/config.sh

mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}/merged/

bash data/create_data_rule/xarg_wrapper.sh

python3 data/merge_pkl.py ${OUTPUT_DIR} ${MAX_PROC}
