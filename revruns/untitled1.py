


from glob import glob
import pandas as pd
import matplotlib as plt



files = glob("*csv")

def box(self, data_dict):
    """
    data_dict = lcoes
    """

    # Setup plot parameters
#         positions = [[0.5, 1.5, 2.5], [4.5, 5.5, 6.5]]
    position = [2, 5, 8]
    legends = ["120m", "140m", "160m"]
    colors = ["peru" ,"green", "skyblue"]
    group_labels = ["20MW Plant", "150MW Plant"]
    legend_patches = []
    for i in range(3):
        patch = mpatches.Patch(color=colors[i], label=legends[i])
        legend_patches.append(patch)

    # Initialize plot
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(0, 10, .5))
    ax.set_title(data_dict["title"])
    plt.ylabel(data_dict["units"])

    # Loop through and make all the boxplots
    for i, ps in enumerate(["20ps", "150ps"]):
        cfdata = data_dict[ps]
        bp = ax.boxplot(cfdata, positions=positions[i], labels=legends,
                        widths = 0.4, showfliers=True,
                        patch_artist=True)
        for y in range(3):
            plt.setp(bp["boxes"][y], color=colors[y], linewidth=3.0)
            plt.setp(bp["fliers"][y], markerfacecolor=colors[y],
                     markeredgecolor=colors[y],linewidth=3.0)
            plt.setp(bp["medians"][y], color="black", linewidth=3.0,
                     alpha=0.2)
            plt.setp(bp["caps"][y * 2], color=colors[y], linewidth=3.0)
            plt.setp(bp["caps"][y * 2 + 1], color=colors[y], linewidth=3.0)
            plt.setp(bp["whiskers"][y * 2], color=colors[y], linewidth=3.0)
            plt.setp(bp["whiskers"][y * 2 + 1], color=colors[y],
                     linewidth=3.0)
    ax.set_xticklabels(group_labels)
    ax.set_xticks([1.5, 5.5])
    plt.legend(handles=legend_patches, title="Hub Height", loc="center",
               bbox_to_anchor=[.5, .1])

    name_bits = data_dict["title"].lower().split()
    fname = "".join(b[0] for b in name_bits) + ".png"
    savepath = os.path.join(self.path, "boxplots", fname)
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath)
