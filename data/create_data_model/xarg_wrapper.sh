source data/create_data_model/config.sh

SHARD_COUNT=0
rm xarg_list.txt
touch xarg_list.txt
PART=0
for GPU_ID in ${GPU_LIST[@]}; do
  echo "${GPU_ID} ${PART}">> xarg_list.txt
  ((PART++))
done
chmod 777 data/create_data_model/create_mask_dataset.sh
xargs -n 2 --max-procs=${MAX_PROC} --arg-file=xarg_list.txt data/create_data_model/create_mask_dataset.sh
rm xarg_list.txt
