import os
import psutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable




def print_memory_usage(tag=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024**2  # in MB
    print(f"[{tag}] Memory usage: {mem:.2f} MB")


def plot_y_vs_x_with_marginals(vals_x, vals_y, label_x, label_y, label_profile, output_filename):
    vals_x = np.asarray(vals_x)
    vals_y = np.asarray(vals_y)

    if np.all(vals_x == vals_x.astype(int)):
        bins_x = np.histogram_bin_edges(vals_x, bins=20)
    else:
        bins_x = np.histogram_bin_edges(vals_x, bins=20)

    if np.all(vals_y == vals_y.astype(int)):
        bins_y = np.histogram_bin_edges(vals_y, bins=20)
    else:
        bins_y = np.histogram_bin_edges(vals_y, bins=20)

    centers_x = (bins_x[:-1] + bins_x[1:]) / 2

    # Compute mean ADC per cm bin
    bin_indices_x = np.digitize(vals_x, bins_x) - 1
    means_y = np.full_like(centers_x, np.nan, dtype=float)
    for i in range(len(centers_x)):
        in_bin = vals_y[bin_indices_x == i]
        if len(in_bin) > 0:
            means_y[i] = np.mean(in_bin)

    # Setup figure with 3 axes
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                           hspace=0.05, wspace=0.05)

    ax_main = plt.subplot(gs[1, 0])
    ax_top = plt.subplot(gs[0, 0], sharex=ax_main)
    ax_right = plt.subplot(gs[1, 1], sharey=ax_main)

    # Show ticks on all 4 sides
    for ax in [ax_main, ax_top, ax_right]:
        ax.tick_params(
            axis='both',
            which='both',
            direction='in',
            top=True,
            bottom=True,
            left=True,
            right=True,
            labelsize=12
        )

    # Force integer ticks on all axes if values are all integers
    if np.all(vals_x == vals_x.astype(int)):
        ax_main.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_top.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    if np.all(vals_y == vals_y.astype(int)):
        ax_main.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_right.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # 2D histogram
    cmap = plt.cm.viridis.copy()
    cmap.set_under("white")
    norm = mcolors.Normalize(vmin=1)

    h = ax_main.hist2d(vals_x, vals_y, bins=(bins_x, bins_y), cmap=cmap, norm=norm)

    # Profile (red dots)
    valid = ~np.isnan(means_y)
    ax_main.scatter(centers_x[valid], means_y[valid], color='red', s=25, label=label_profile, zorder=10)
    ax_main.legend(fontsize=15)
    ax_main.tick_params(labelsize=15)

    # 1D histograms
    ax_top.hist(vals_x, bins=bins_x, color='gray')
    ax_right.hist(vals_y, bins=bins_y, color='gray', orientation='horizontal')

    # Clean ticks and labels
    ax_top.tick_params(axis='x', labelbottom=False)
    ax_right.tick_params(axis='y', labelleft=False)

    ax_main.set_xlabel(label_x, fontsize=19, loc='right', labelpad=10)
    ax_main.set_ylabel(label_y, fontsize=19, loc='top', labelpad=10)

    ax_main.grid(True)
    plt.savefig(output_filename)
    print(f"Saved 2-d plot with marginals {output_filename}")
    plt.close()