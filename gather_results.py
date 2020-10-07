import os
import sys


dir_path = sys.argv[1]

tot = 0
seeds = [13, 43, 83, 181, 271, 347, 433, 659, 727, 859]
for seed in seeds:
    path = os.path.join(dir_path, str(seed), "test_results.txt")
    with open(path, "r") as f:
        acc = float(f.readlines()[0].strip().split()[-1])
        for l in f.readlines():
            print(l)
    tot += acc

print("Gathered results on different random seeds. Average Acc. : ")
print(tot / len(seeds))
