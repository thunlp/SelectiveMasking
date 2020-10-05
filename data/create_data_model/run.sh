source data/create_data_model/config.sh

mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}/model/
mkdir -p ${OUTPUT_DIR}/rand/
mkdir -p ${OUTPUT_DIR}/model/merged
mkdir -p ${OUTPUT_DIR}/rand/merged

source data/create_data_model/xarg_wrapper.sh

python3 data/merge_hdf5.py ${OUTPUT_DIR}/model/ ${MAX_PROC}
python3 data/merge_hdf5.py ${OUTPUT_DIR}/rand/ ${MAX_PROC}
