import os
import sys


dir_path = sys.argv[1]
step = sys.argv[2]

tot = 0
for seed in [13, 43, 83, 181, 271, 347, 433, 659, 727, 859]:
    path = os.path.join(dir_path, str(seed), "ckpt_{}".format(str(step)), "test_eval_results.txt")
    with open(path, "r") as f:
        acc = float(f.readlines()[0].strip().split()[-1])
        for l in f.readlines():
            print(l)
    tot += acc

print("avg: ")
print(tot / 10)