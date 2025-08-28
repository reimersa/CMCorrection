import os
import psutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from typing import Union

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable




def print_memory_usage(tag=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024**2  # in MB
    print(f"[{tag}] Memory usage: {mem:.2f} MB")


def overlay_profiles(vals_x, list_of_vals_y, label_x, label_y, labels_profiles, output_filename, nbins_x=20, ratio_to = None):
    if not len(list_of_vals_y) == len(labels_profiles):
        raise ValueError(f"Number of profiles to plot ({len(list_of_vals_y)}) does not match number of profile labels ({len(labels_profiles)})")
    vals_x = np.asarray(vals_x)
    list_of_vals_y = [np.asarray(v) for v in list_of_vals_y]

    bins_x = np.histogram_bin_edges(vals_x, bins=nbins_x)
    centers_x = (bins_x[:-1] + bins_x[1:]) / 2
    bin_indices_x = np.digitize(vals_x, bins_x) - 1

    list_of_means_y = [np.full_like(centers_x, np.nan, dtype=float) for v in list_of_vals_y]
    for i in range(len(centers_x)):
        for j in range(len(list_of_means_y)):
            in_bin = list_of_vals_y[j][bin_indices_x == i]
            if len(in_bin) > 0:
                list_of_means_y[j][i] = np.mean(in_bin)

    
    if ratio_to is None:
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        axes = (ax,)
    else:
        fig = plt.figure(figsize=(6.8, 6.2))
        gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax  = fig.add_subplot(gs[0])
        axr = fig.add_subplot(gs[1], sharex=ax)
        axes = (ax, axr)

    # Top panel: overlay means
    for m, lbl in zip(list_of_means_y, labels_profiles):
        valid = ~np.isnan(m)
        ax.plot(centers_x[valid], m[valid], label=lbl, linewidth=2, marker="o", markersize=4)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if ratio_to is not None:
        # Hide xticklabels on top panel (shared axis)
        plt.setp(ax.get_xticklabels(), visible=False)

        # Ratio panel
        base = list_of_means_y[int(ratio_to)]
        # Build ratios with safe division: mask where base is nan or ~0
        eps = 0.0
        denom_mask = (~np.isnan(base)) & (np.abs(base) > eps)

        for m, lbl in zip(list_of_means_y, labels_profiles):
            ratio = np.full_like(base, np.nan, dtype=float)
            ok = denom_mask & (~np.isnan(m))
            ratio[ok] = m[ok] / base[ok]
            valid = ~np.isnan(ratio)
            axr.plot(centers_x[valid], ratio[valid], linewidth=1.8, marker=".", markersize=3, label=f"{lbl} / {labels_profiles[int(ratio_to)]}")

        axr.axhline(1.0, color="k", lw=1, ls="--", alpha=0.6)
        axr.set_xlabel(label_x)
        axr.set_ylabel("ratio")
        axr.grid(True, alpha=0.3)
        axr.set_ylim(0.8, 1.2)

    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Saved overlay of means: {output_filename}")
    plt.close()


def plot_y_vs_x_with_marginals(vals_x, vals_y, label_x, label_y, label_profile, output_filename, nbins_x=20, nbins_y=20, yrange=[-20, 20]):
    vals_x = np.asarray(vals_x)
    vals_y = np.asarray(vals_y)

    bins_x = np.histogram_bin_edges(vals_x, bins=nbins_x)
    bins_y = np.histogram_bin_edges(yrange, bins=nbins_y)

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
    plt.subplots_adjust(left=0.10, right=0.99, bottom=0.10, top=0.97)
    plt.savefig(output_filename)
    print(f"Saved 2-d plot with marginals {output_filename}")
    plt.close()

def plot_covariance(df, nch_per_erx, title, xtitle, ytitle, ztitle, output_filename, zrange=(-1., 1.)):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot 2D heatmap
    im = ax.pcolormesh(
        df.columns,  # x bin edges
        df.index,    # y bin edges
        df.values,   # 2D values
        shading='auto',
        cmap='coolwarm',   # or 'RdBu', 'coolwarm', 'plasma', etc.,
        vmin=zrange[0],
        vmax=zrange[1]
    )

    # Draw dashed lines every `nch_per_erx` channels
    n_channels = df.shape[0]
    for i in range(nch_per_erx, n_channels, nch_per_erx):
        ax.axhline(i-0.5, color='black', linestyle='--', linewidth=0.7)
        ax.axvline(i-0.5, color='black', linestyle='--', linewidth=0.7)

    ticks = np.arange(0, n_channels, nch_per_erx)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)

    # Labels and styling
    ax.set_title(title)
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(ztitle)

    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Saved 2-d plot {output_filename}")
    plt.close()


def plot_hist_single(x: np.ndarray, bins: Union[int, np.ndarray], color: str, xlabel: str, title: str, outpath: str, show_mean_line: bool = True) -> None:
    edges = np.histogram_bin_edges(x, bins=bins) if isinstance(bins, int) else bins
    plt.figure(figsize=(8, 5))
    plt.hist(x, bins=edges, color=color)
    if show_mean_line:
        mu = float(np.mean(x))
        plt.axvline(mu, color="k", ls="--", label=f"mean = {mu:.3%}")
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel("Number of channels")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath); plt.close()

def plot_hist_overlay_pair(a: np.ndarray, b: np.ndarray, bins: Union[int, np.ndarray], label_a: str, label_b: str, color_a: str, color_b: str, xlabel: str, title: str, outpath: str) -> None:
    # common bins from both series if bins is an int
    if isinstance(bins, int):
        edges = np.histogram_bin_edges(np.concatenate([a, b]), bins=bins)
    else:
        edges = bins
    plt.figure(figsize=(8, 5))
    plt.hist(a, bins=edges, alpha=0.6, label=label_a, color=color_a)
    plt.hist(b, bins=edges, alpha=0.6, label=label_b, color=color_b)
    plt.xlabel(xlabel)
    plt.ylabel("Number of channels")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath); plt.close()

def truncated_rms(x: np.ndarray, fraction: float = 1.0) -> float:
    x = np.asarray(x).ravel()
    if x.size == 0: 
        return np.nan
    if not (0 < fraction <= 1): 
        raise ValueError("fraction must be in (0, 1].")
    if fraction == 1.0: 
        return np.sqrt(np.nanmean(x**2))

    lo = (1.0 - fraction) / 2.0
    hi = 1.0 - lo

    vmin, vmax = np.quantile(x, [lo, hi])
    sel = x[(x >= vmin) & (x <= vmax)]
    
    return np.sqrt(np.nanmean(sel**2)) if sel.size else np.nan