NUM=$1
DIR1=${HOME}/SelectiveMasking/data/amazon_model_gen/hdf5_shards
DIR2=${HOME}/SelectiveMasking/data/yelp_mask_info/hdf5_shards
OUTPUT_DIR=${HOME}/SelectiveMasking/data/merged_amazon_yelp/hdf5_shards

python3 ${HOME}/SelectiveMasking/merge_hdf5.py ${DIR1}/${NUM}.hdf5 $DIR2/${NUM}.hdf5 ${OUTPUT_DIR}/${NUM}.hdf5