import h5py
import sys
import numpy as np
import collections
from tqdm import tqdm

filename1 = sys.argv[1]
filename2 = sys.argv[2]
output_file = sys.argv[3]

f1 = h5py.File(filename1, "r")
f2 = h5py.File(filename2, "r")

num_inst1 = f1["input_ids"].shape[0]
num_inst2 = f2["input_ids"].shape[0]

max_seq_length = f1["input_ids"].shape[1]
max_predictions_per_seq = f1["masked_lm_positions"].shape[1]
num_instances = num_inst1 + num_inst2

features = collections.OrderedDict()
features["input_ids"] = np.zeros([num_instances, max_seq_length], dtype="int32")
features["input_mask"] = np.zeros([num_instances, max_seq_length], dtype="int32")
features["segment_ids"] = np.zeros([num_instances, max_seq_length], dtype="int32")
features["masked_lm_positions"] = np.zeros([num_instances, max_predictions_per_seq], dtype="int32")
features["masked_lm_ids"] = np.zeros([num_instances, max_predictions_per_seq], dtype="int32")
features["next_sentence_labels"] = np.zeros(num_instances, dtype="int32")

L = []

for i in tqdm(range(num_inst1)):
    L.append((f1["input_ids"][i, :], f1["input_mask"][i, :], f1["segment_ids"][i, :], 
             f1["masked_lm_positions"][i, :], f1["masked_lm_ids"][i, :], f1["next_sentence_labels"][i]))

for i in tqdm(range(num_inst2)):
    L.append((f2["input_ids"][i, :], f2["input_mask"][i, :], f2["segment_ids"][i, :],
             f2["masked_lm_positions"][i, :], f2["masked_lm_ids"][i, :], f2["next_sentence_labels"][i]))

np.random.shuffle(L)
for i, instance in enumerate(tqdm(L)):
    features["input_ids"][i] = instance[0]
    features["input_mask"][i] = instance[1]
    features["segment_ids"][i] = instance[2]
    features["masked_lm_positions"][i] = instance[3]
    features["masked_lm_ids"][i] = instance[4]
    features["next_sentence_labels"][i] = instance[5]

print("saving data")
f = h5py.File(output_file, 'w')
f.create_dataset("input_ids", data=features["input_ids"], dtype='i4', compression='gzip')
f.create_dataset("input_mask", data=features["input_mask"], dtype='i1', compression='gzip')
f.create_dataset("segment_ids", data=features["segment_ids"], dtype='i1', compression='gzip')
f.create_dataset("masked_lm_positions", data=features["masked_lm_positions"], dtype='i4', compression='gzip')
f.create_dataset("masked_lm_ids", data=features["masked_lm_ids"], dtype='i4', compression='gzip')
f.create_dataset("next_sentence_labels", data=features["next_sentence_labels"], dtype='i1', compression='gzip')
f.flush()
f.close()
