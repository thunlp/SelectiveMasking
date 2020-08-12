import numpy as np
import matplotlib.pyplot as plt


font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 25,
         }

font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 35,
         }

fig = plt.figure(figsize=(10, 10))
sub = fig.add_subplot(111)

shiftY = 0.4
shiftX = 35
ms = 11

fontsize1 = 28
fontsize2 = 35

model_steps_30 = [300, 480]
model_score_30 = [81.34, 87.19]
l2, = sub.plot(model_steps_30, model_score_30, 'b^-', ms=ms)
sub.text(model_steps_30[-1] + shiftX, model_score_30[-1], "300k\nTask", ha='center', va='center', fontsize=fontsize1)

rand_steps_30 = [300, 488]
rand_score_30 = [81.34, 85.85]
l3, = sub.plot(rand_steps_30, rand_score_30, 'gs-', ms=ms)
sub.text(rand_steps_30[-1], rand_score_30[-1] + shiftY, "300k\nRand.", ha='center', va='center', fontsize=fontsize1)

model_steps_20 = [200, 380]
model_score_20 = [80.99, 86.43]
sub.plot(model_steps_20, model_score_20, 'b^-', ms=ms)
sub.text(model_steps_20[-1] + shiftX, model_score_20[-1], "200k\nTask", ha='center', va='center', fontsize=fontsize1)

rand_steps_20 = [200, 384]
rand_score_20 = [80.99, 85.22]
sub.plot(rand_steps_20, rand_score_20, 'gs-', ms=ms)
sub.text(rand_steps_20[-1], rand_score_20[-1] + shiftY, "200k\nRand.", ha='center', va='center', fontsize=fontsize1)

model_steps_10 = [100, 264]
model_score_10 = [79.7, 85.32]
sub.plot(model_steps_10, model_score_10, 'b^-', ms=ms)
sub.text(model_steps_10[-1], model_score_10[-1] + shiftY, "100k\nTask", ha='center', va='center', fontsize=fontsize1)

rand_steps_10 = [100, 272]
rand_score_10 = [79.7, 84.57]
sub.plot(rand_steps_10, rand_score_10, 'gs-', ms=ms)
sub.text(rand_steps_10[-1] + shiftX, rand_score_10[-1] + shiftY, "100k\nRand.", ha='center', va='center', fontsize=fontsize1)

main_steps = [100, 200, 300]
main_score = [79.7, 80.99, 81.34]
l1, = sub.plot(main_steps, main_score, 'ro-', ms=ms)

for step, score in zip(main_steps, main_score):
    sub.text(step + shiftX / 5 * 4, score - shiftY / 4, str(step) + "k", ha='center', va='center', fontsize=fontsize1)
sub.hlines(87.37, 100, 500, colors="gray", linestyles="dashed", linewidth=3)
sub.text(210, 87, "Fully-trained (1M steps)", ha='center', va='center', fontsize=27)
sub.text(48, 87.37, "87.4-", ha='center', va='center', fontsize=fontsize2)

plt.grid()
plt.tick_params(labelsize=fontsize2)

plt.xlabel("k Steps", font2)
plt.ylabel("Acc.(%)", font2)

plt.legend(handles=[l1, l2, l3], labels=['General Pre-train', 'Selective Mask', 'Random Mask'], loc='lower right', prop=font1)
plt.savefig("../images/mr_amazon.pdf", foramt="pdf")
