import matplotlib.pyplot as plt
import os

plt.rcParams["font.size"] = 28
plt.rcParams["figure.autolayout"] = True

if __name__ == '__main__':
    plt.figure(0)
    acq_func_list = ["Ours", "MELK", "MILE", "RMILE", "Str", "US"]
    color_list = ["r", "g", "b", "cyan", "purple", "orange"]

    save_dir = "result"
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots()

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    colors = ["red", "blue", "green", "yellow", "magenta", "olive"]
    for a, acq_func_name in enumerate(acq_func_list):
        for s in range(2):
            if s == 0:
                if acq_func_name == "Ours":
                    ax.scatter([], [], c=color_list[a], marker="o", s=130, label=f"Ours", edgecolors="k")
                else:
                    ax.scatter([], [], c=color_list[a], marker="o", s=130, edgecolors="k")
            if s == 1:
                ax.scatter([], [], c=color_list[a], marker=",", s=130, label=f"FS({acq_func_list[a]})", edgecolors="k")
            else:
                ax.scatter([], [], c=color_list[a], marker="^", s=130, label=f"FC({acq_func_list[a]})", edgecolors="k")

    for a, acq_func_name in enumerate(acq_func_list):
        ax.plot([], [], c=color_list[a], label=acq_func_name, zorder=len(acq_func_list) - a, lw=2, alpha=0.5)

    legend = ax.legend(frameon=False, handletextpad=0, ncol=3, fontsize=24, columnspacing=1)
    # plt.legend(borderaxespad=0, ncol=2, framealpha=0.7, handlelength=0.7, loc="lower right")

    legend_fig = legend.figure
    legend_fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(legend_fig.dpi_scale_trans.inverted())

    plt.savefig(f"{save_dir}/legend.pdf", bbox_inches=bbox)
