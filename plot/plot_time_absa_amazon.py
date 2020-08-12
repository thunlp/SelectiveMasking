import numpy as np
import matplotlib.pyplot as plt

fracs = [1244, 136, 829, 2160]
labels = ["GenePT", "Selective Masking", "TaskPT", "Saved Cost"]

plt.figure(figsize=(9, 9))

plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)

plt.axes(aspect=1)
plt.pie(
    x=fracs,
    labels=labels,
    startangle=90, 
    # colors=["lightskyblue", "orange", "cyan", "lightcoral", "gold", "lightgreen", "white"],
    colors=["lightskyblue", "gold", "lightgreen", "white"],
    wedgeprops={'linewidth': 0.5, 'edgecolor': "black"},
    explode=[0, 0, 0, 0.06],
    shadow=True,
    labeldistance=10,
    radius=1,
    autopct='%3.1f %%',
    pctdistance=0.8,
    textprops={
        'size': 16
    }
    )

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 11.5,
         }
plt.legend(loc="upper right", prop=font1)
plt.savefig("../images/time_absa_amazon.pdf", format="pdf")
