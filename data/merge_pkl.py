import pickle
import sys
import os
import random

origin_dir = sys.argv[1]
num_files = int(sys.argv[2])

L = []

dev_rate = 0.1

for i in range(num_files):
    filename = os.path.join(origin_dir, "{}.pkl".format(i))
    print(filename)
    with open(filename, "rb") as f:
        L.extend(pickle.load(f))

all_data_size = len(L)

train_data = L[0: int((1 - dev_rate) * all_data_size)]
dev_data = L[int((1 - dev_rate) * all_data_size):]

with open(os.path.join(origin_dir, "merged", "train.pkl"), "wb") as f:
    pickle.dump(train_data, f)

with open(os.path.join(origin_dir, "merged", "valid.pkl"), "wb") as f:
    pickle.dump(dev_data, f)
