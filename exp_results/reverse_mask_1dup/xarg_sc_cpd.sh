source utils/config.sh

SHARD_COUNT=0
rm -rf xarg_list.txt
touch xarg_list.txt
for ((SHARD_COUNT=0; SHARD_COUNT<$N_PROCS_PREPROCESS; SHARD_COUNT++)); do
  echo "${SHARD_COUNT}">> xarg_list.txt
done
chmod 777 ./sc_cpd.sh
xargs -n 1 --max-procs=${N_PROCS_PREPROCESS} --arg-file=xarg_list.txt $HOME/nvidia-bert/data/sc_cpd.sh
rm xarg_list.txt
