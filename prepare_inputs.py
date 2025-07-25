import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal.*")
import uproot
import pandas as pd
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from copy import copy
import os
from sklearn.model_selection import train_test_split

import utils


def main():

    # Names of the modules for which we want to produce inputs
    modulenames = ["ML_F3W_WXIH0190", "ML_F3W_WXIH0191", "ML_F3W_WXIH0192", "ML_F3W_WXIH0193", "ML_F3W_WXIH0194", "ML_F3W_WXIH0196", "ML_F3W_WXIH0197", "ML_F3W_WXIH0198"]





    ### No need to change things below here -------------------------

    for modulename in modulenames:
        # Define output folders
        plotfolder = f"plots/inputs/{modulename}"
        inputfolder = f"/eos/user/{os.getenv('USER')[0]}/{os.getenv('USER')}/hgcal/dnn_inputs/{modulename}"
        os.makedirs(name=plotfolder, exist_ok=True)
        os.makedirs(name=inputfolder, exist_ok=True)

        # Open file and load tree
        infilename = f"/eos/user/a/areimers/hgcal/output/histofiller/input_features_{modulename}.root"
        infile = uproot.open(infilename)
        tree = infile["InputFeatures"]
    
        # Load tree into df
        df = tree.arrays(library="pd")
        for idx_erx in range(12):
            df.loc[df["nerx"] <= idx_erx, f"cm_erx{idx_erx:02}"] = 0
    
        # Define input variables, everything we need from the ROOT files
        input_cols = [f'cm_erx{idxerx:02}' for idxerx in range(12)] + ['erx', 'chadc', 'nchadc']
    
        # Expand to flat arrays
        inputs, targets, colnames = expand_inputs_and_targets(df, input_cols=input_cols, selected_indices=range(222))
    
        # Preprocess flat arrays (center them and save means)
        inputs, targets, inputs_mean, targets_mean, chadc, eventid = preprocess_inputs_targets(inputs, targets, foldername=inputfolder)
        for inputname in colnames:
            labelname = inputname.replace("cm_erx", "CM e-Rx ")
            utils.plot_y_vs_x_with_marginals(vals_x=inputs[inputname], vals_y=targets["adc"], label_x=labelname, label_y="Measured ADC", label_profile="profile: mean measured ADC", output_filename=f"{plotfolder}/ADC_vs_{inputname}_mean0.pdf")
    
        # Perform train/val split
        all_indices = np.arange(len(inputs))
        train_indices, val_indices = train_test_split(
            all_indices,
            test_size=0.2,
            random_state=42,
            shuffle=True
        )

        # Drop columns that should not be used as inputs (but were loaded because needed for something else)
        inputs = inputs.drop(columns=["event_id"])
    
        # Convert inputs and targets to numpy
        X = inputs.to_numpy()
        y = targets.to_numpy()
    
        # Apply train/val split
        X_train = X[train_indices]
        X_val = X[val_indices]
        y_train = y[train_indices]
        y_val = y[val_indices]

        # Write to files
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
    """Some inputs are not channel-specific, but module-specific. These need to be repeated to have the same format as per-channel inputs."""

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
    """Subtract average ADC per channel (= pedestal correction), also center all other variables around 0 for numerical stability."""

    # Center each input column (scalar + per-channel inputs)
    inputs_mean = np.mean(inputs, axis=0)
    inputs_centered = inputs - inputs_mean

    # Use chadc to get per-channel mean for each ADC value
    chadc = inputs["chadc"].astype(int)
    eventid = inputs["event_id"].astype(int)
    adc_vals = targets["adc"]

    # Compute per-channel means
    per_channel_means = adc_vals.groupby(chadc).mean()

    # Subtract per-channel means
    targets_centered = adc_vals - chadc.map(per_channel_means)
    targets_centered_df = pd.DataFrame({"adc": targets_centered})

    return inputs_centered.astype(np.float32), targets_centered_df.astype(np.float32), inputs_mean, per_channel_means.to_numpy(), chadc.to_numpy(), eventid.to_numpy()


if __name__ == '__main__':
  main()
