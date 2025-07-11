import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal.*")
import uproot
import pandas as pd
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import numpy as np
from copy import copy
import os
from sklearn.model_selection import train_test_split


def main():
    # Open the ROOT file and tree
    # modulenames = ["ML_F3W_WXIH0190", "ML_F3W_WXIH0191"]
    modulenames = ["ML_F3W_WXIH0190", "ML_F3W_WXIH0191", "ML_F3W_WXIH0192", "ML_F3W_WXIH0193", "ML_F3W_WXIH0194", "ML_F3W_WXIH0195", "ML_F3W_WXIH0196", "ML_F3W_WXIH0197", "ML_F3W_WXIH0198"]

    for modulename in modulenames:
        plotfolder = f"plots/inputs/{modulename}"
        inputfolder = f"/eos/user/a/areimers/hgcal/dnn_inputs/{modulename}"
        os.makedirs(name=plotfolder, exist_ok=True)
        os.makedirs(name=inputfolder, exist_ok=True)
        infilename = f"../output/histofiller/input_features_{modulename}.root"
        file = uproot.open(infilename)
        tree = file["InputFeatures"]
    
        # Load tree into df
        df = tree.arrays(library="pd")
        for idx_erx in range(12):
            df.loc[df["nerx"] <= idx_erx, f"cm_erx{idx_erx:02}"] = 0
    
        input_cols = [f'cm_erx{idxerx:02}' for idxerx in range(12)] + ['erx', 'chadc', 'nchadc']
    
        # Expand to flat arrays
        # inputs, targets = expand_inputs_and_targets(df, input_cols=input_cols, selected_indices=range(37))
        inputs, targets, colnames = expand_inputs_and_targets(df, input_cols=input_cols, selected_indices=range(222))
        print(f"after expanding inputs, the column names are: {colnames}")
        # inputs, targets = expand_inputs_and_targets(df, input_cols=input_cols, selected_indices=range(1))
        # for inputname in colnames:
        #     plot_y_vs_x_with_marginals(vals_x=inputs[inputname], vals_y=targets["adc"], label_x=inputname, label_y="ADC channels 0-222", label_profile="ADC profile", output_filename=f"{plotfolder}/ADC_vs_{inputname}.pdf")
    
        # Preprocess flat arrays (center them and save means)
        inputs, targets, inputs_mean, targets_mean, chadc, eventid = preprocess_inputs_targets(inputs, targets, foldername=inputfolder)
        for inputname in colnames:
            labelname = inputname.replace("cm_erx", "CM e-Rx ")
            plot_y_vs_x_with_marginals(vals_x=inputs[inputname], vals_y=targets["adc"], label_x=labelname, label_y="Measured ADC", label_profile="profile: mean measured ADC", output_filename=f"{plotfolder}/ADC_vs_{inputname}_mean0.pdf")
    
        # Perform train/val split
        all_indices = np.arange(len(inputs))
        train_indices, val_indices = train_test_split(
            all_indices,
            test_size=0.2,
            random_state=42,
            shuffle=True
        )

        inputs = inputs.drop(columns=["event_id"])
    
        X = inputs.to_numpy()
        y = targets.to_numpy()
    
        X_train = X[train_indices]
        X_val = X[val_indices]
        y_train = y[train_indices]
        y_val = y[val_indices]


    
        np.save(f"{inputfolder}/inputs_train.npy", X_train)
        np.save(f"{inputfolder}/inputs_val.npy", X_val)
        np.save(f"{inputfolder}/targets_train.npy", y_train)
        np.save(f"{inputfolder}/targets_val.npy", y_val)
        np.save(f"{inputfolder}/indices_train.npy", train_indices)
        np.save(f"{inputfolder}/indices_val.npy", val_indices)
        np.save(f"{inputfolder}/inputs.npy", inputs.to_numpy())
        np.save(f"{inputfolder}/targets.npy", targets.to_numpy())
        np.save(f"{inputfolder}/inputs_mean.npy", inputs_mean)
        np.save(f"{inputfolder}/targets_mean.npy", targets_mean)
        np.save(f"{inputfolder}/chadc.npy", chadc)
        np.save(f"{inputfolder}/eventid.npy", eventid)
        print(f"--> Wrote DNN inputs/targets to {inputfolder}")


def expand_inputs_and_targets(df, input_cols, selected_indices):
    # --- Separate per-channel inputs from scalar inputs ---
    scalar_cols = [col for col in input_cols if col not in ["chadc", "erx"]]

    # ---------- NEW: build event_id flat array --------------------------
    n_ch   = len(selected_indices)               # 37 or 222 or whatever user wants
    n_evt  = len(df)                             # one row per event
    event_id_flat = np.repeat(np.arange(n_evt, dtype=np.int32), n_ch)

    # --- Extract and repeat scalar inputs ---
    scalar_inputs = df[scalar_cols].astype(np.float32)
    scalar_inputs_repeated = pd.DataFrame(
        np.repeat(scalar_inputs.values, len(selected_indices), axis=0),
        columns=scalar_cols
    )

    # --- Extract and slice per-channel inputs ---
    per_channel_data = {}
    for col in ["chadc", "erx"]:
        matrix = np.vstack(df[col].to_numpy())[:, selected_indices]
        flat = matrix.flatten().astype(np.float32)
        per_channel_data[col] = flat

    # --- Combine all into a single DataFrame ---
    inputs_df = scalar_inputs_repeated.copy()
    for col, values in per_channel_data.items():
        if col in ["erx"]:
            # Convert to category for consistent one-hot encoding
            values = pd.Series(values, name=col).astype(int)
            formatted_labels = [f"{val:02}" for val in sorted(values.unique())]
            cat_type = CategoricalDtype(categories=sorted(values.unique()), ordered=False)
            values = values.astype(cat_type)
            onehot = pd.get_dummies(values)
            onehot = onehot.astype(np.uint8)
            onehot.columns = [f"{col}_{label}" for label in formatted_labels]
            inputs_df = pd.concat([inputs_df, onehot], axis=1)
        else:
            inputs_df[col] = values

    # ---------- NEW: attach the event_id column ------------------------
    inputs_df["event_id"] = event_id_flat       # plain integer

    # --- Targets: extract and flatten selected ADC channels ---
    adc_matrix = np.stack(df["adc"].to_numpy())[:, selected_indices]
    targets_flat = adc_matrix.flatten().astype(np.float32)
    targets_df = pd.DataFrame({"adc": targets_flat})

    return inputs_df, targets_df, inputs_df.columns.tolist()


def preprocess_inputs_targets(inputs, targets, foldername):
    # Center each input column (scalar + per-channel inputs)
    inputs_mean = np.mean(inputs, axis=0)
    inputs_centered = inputs - inputs_mean

    # Use chadc to get per-channel mean for each ADC value
    # print(inputs["chadc"])
    chadc = inputs["chadc"].astype(int)
    eventid = inputs["event_id"].astype(int)
    # print(chadc)
    adc_vals = targets["adc"]

    # Compute per-channel means
    per_channel_means = adc_vals.groupby(chadc).mean()

    # Subtract per-channel means
    targets_centered = adc_vals - chadc.map(per_channel_means)
    targets_centered_df = pd.DataFrame({"adc": targets_centered})

    # Save inputs and targets
    # np.save(f"{foldername}/inputs_erx00_mean.npy", inputs_mean)
    # np.save(f"{foldername}/targets_erx00_mean.npy", per_channel_means.to_numpy())
    # np.save(f"{foldername}/inputs_mean.npy", inputs_mean)
    # np.save(f"{foldername}/targets_mean.npy", per_channel_means.to_numpy())
    # np.save(f"{foldername}/inputs_ch000_mean.npy", inputs_mean)
    # np.save(f"{foldername}/targets_ch000_mean.npy", per_channel_means.to_numpy())

    # Also save chadc for use during evaluation (e.g., undoing mean subtraction)
    # np.save(f"{foldername}/chadc.npy", chadc.to_numpy())

    return inputs_centered.astype(np.float32), targets_centered_df.astype(np.float32), inputs_mean, per_channel_means.to_numpy(), chadc.to_numpy(), eventid.to_numpy()


def plot_1d_histogram(data, output_filename, bins='auto', xlabel=None, ylabel="Entries", integer_ticks=True):

    # Determine what type of input we have
    if isinstance(data, str):
        raise ValueError("Please pass the actual data or a column Series, not a string column name.")
    elif isinstance(data, pd.Series):
        values = data.to_numpy()
        xlabel = xlabel or data.name
    else:
        values = np.array(data)
        if xlabel is None:
            xlabel = "Value"

    # Auto-binning for integers
    if bins == 'auto':
        if np.issubdtype(values.dtype, np.integer):
            min_val, max_val = values.min(), values.max()
            bins = np.arange(min_val, max_val + 2) - 0.5
        else:
            bins = 50

    # Plot
    plt.figure(figsize=(8, 6))
    plt.hist(values, bins=bins, color="steelblue")

    ax = plt.gca()
    ax.set_xlabel(xlabel, fontsize=16, loc="right", labelpad=10)
    ax.set_ylabel(ylabel, fontsize=16, loc="top", labelpad=10)
    ax.tick_params(axis='both', labelsize=12)

    if integer_ticks:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Saved 1-d plot {output_filename}")
    plt.close()


def plot_y_vs_x_with_marginals(vals_x, vals_y, label_x, label_y, label_profile, output_filename):
    vals_x = np.asarray(vals_x)
    vals_y = np.asarray(vals_y)

    if np.all(vals_x == vals_x.astype(int)):
        # bins_x = np.arange(int(vals_x.min()), int(vals_x.max()) + 2) - 0.5
        bins_x = np.histogram_bin_edges(vals_x, bins=20)
    else:
        bins_x = np.histogram_bin_edges(vals_x, bins=20)

    if np.all(vals_y == vals_y.astype(int)):
        # bins_y = np.arange(int(vals_y.min()), int(vals_y.max()) + 2) - 0.5
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
    # fig.tight_layout()
    plt.savefig(output_filename)
    print(f"Saved 2-d plot with marginals {output_filename}")
    plt.close()


def plot_adc_vs_cm_profile(df, adc_channel, cm_column, output_filename):

    # Extract data
    adc_ch0 = np.array([v[adc_channel] for v in df["adc"]])
    vals_x = df[cm_column].to_numpy()

    # Define binning
    bins_y = np.arange(adc_ch0.min(), adc_ch0.max() + 2) - 0.5
    bins_x = np.arange(vals_x.min(), vals_x.max() + 2) - 0.5
    cm_bin_centers = (bins_x[:-1] + bins_x[1:]) / 2

    # Digitize cm values to find which adc values fall into which cm bin
    bin_indices_x = np.digitize(vals_x, bins_x) - 1

    # Compute mean adc in each cm bin
    means_y = np.full_like(cm_bin_centers, np.nan, dtype=float)
    for i in range(len(cm_bin_centers)):
        values_in_bin = adc_ch0[bin_indices_x == i]
        if len(values_in_bin) > 0:
            means_y[i] = values_in_bin.mean()

    # 2D histogram
    cmap = plt.cm.viridis.copy()
    cmap.set_under("white")
    norm = mcolors.Normalize(vmin=1)

    plt.figure(figsize=(8, 6))
    plt.hist2d(vals_x, adc_ch0, bins=(bins_x, bins_y), cmap=cmap, norm=norm)

    # Axis labels
    plt.xlabel(cm_column, fontsize=16, loc="right", labelpad=10)
    plt.ylabel("ADC in Channel 0", fontsize=16, loc="top", labelpad=10)

    # Colorbar
    cbar = plt.colorbar()
    cbar.set_label("Entries", fontsize=14)

    # Overlay profile as red scatter points
    valid = ~np.isnan(means_y)
    plt.scatter(
        cm_bin_centers[valid],
        means_y[valid],
        color="red",
        label="ADC profile",
        s=25,
        zorder=10
    )
    plt.legend(fontsize=12)

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Saved 2-d plot {output_filename}")
    plt.close()


if __name__ == '__main__':
  main()
