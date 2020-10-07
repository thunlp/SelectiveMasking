source data/create_data_model/config.sh

mkdir -p ${OUTPUT_DIR}/model/merged/dev
mkdir -p ${OUTPUT_DIR}/rand/merged/dev

bash data/create_data_model/xarg_wrapper.sh

python3 data/merge_hdf5.py ${OUTPUT_DIR}/model/ ${MAX_PROC}
python3 data/merge_hdf5.py ${OUTPUT_DIR}/rand/ ${MAX_PROC}
