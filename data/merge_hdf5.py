import h5py
import sys
import os
import numpy as np
import collections
from tqdm import tqdm

origin_dir = sys.argv[1]
num_files = int(sys.argv[2])

features_trn, features_dev = collections.OrderedDict(), collections.OrderedDict()
num_instances, max_seq_length, max_predictions_per_seq = 0, 0, 0
data = []

dev_rate = 0.1

for i in range(num_files):
    filename = os.path.join(origin_dir, "{}.hdf5".format(i))
    with h5py.File(filename, "r") as f:
        num_inst = f["input_ids"].shape[0]
        max_seq_length = f["input_ids"].shape[1]
        max_predictions_per_seq = f["masked_lm_positions"].shape[1]
        num_instances += num_inst
        for k in tqdm(range(num_inst), desc="loading {}".format(filename)):
            data.append({
                "input_ids": f["input_ids"][i, :], 
                "input_mask": f["input_mask"][i, :], 
                "segment_ids": f["segment_ids"][i, :], 
                "masked_lm_positions": f["masked_lm_positions"][i, :], 
                "masked_lm_ids": f["masked_lm_ids"][i, :], 
                "next_sentence_labels": f["next_sentence_labels"][i]
            })

train_data = data[:int((1 - dev_rate) * num_instances)]
dev_data = data[int((1 - dev_rate) * num_instances):]

for feat, d, name in [(features_trn, train_data, "train.hdf5"), (features_dev, dev_data, "dev/dev.hdf5")]:
    feat["input_ids"] = np.zeros([num_instances, max_seq_length], dtype="int32")
    feat["input_mask"] = np.zeros([num_instances, max_seq_length], dtype="int32")
    feat["segment_ids"] = np.zeros([num_instances, max_seq_length], dtype="int32")
    feat["masked_lm_positions"] = np.zeros([num_instances, max_predictions_per_seq], dtype="int32")
    feat["masked_lm_ids"] = np.zeros([num_instances, max_predictions_per_seq], dtype="int32")
    feat["next_sentence_labels"] = np.zeros(num_instances, dtype="int32")

    for i, inst in enumerate(tqdm(d, desc="train data")):
        for key in feat:
            feat[key][i] = inst[key]

    f = h5py.File(os.path.join(origin_dir, "merged", name), 'w')
    f.create_dataset("input_ids", data=feat["input_ids"], dtype='i4', compression='gzip')
    f.create_dataset("input_mask", data=feat["input_mask"], dtype='i1', compression='gzip')
    f.create_dataset("segment_ids", data=feat["segment_ids"], dtype='i1', compression='gzip')
    f.create_dataset("masked_lm_positions", data=feat["masked_lm_positions"], dtype='i4', compression='gzip')
    f.create_dataset("masked_lm_ids", data=feat["masked_lm_ids"], dtype='i4', compression='gzip')
    f.create_dataset("next_sentence_labels", data=feat["next_sentence_labels"], dtype='i1', compression='gzip')
    f.flush()
    f.close()
