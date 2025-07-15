import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import math

import utils
import models


def main():

    # --- Define paths ---
    # modulename_model = "ML_F3W_WXIH0190_ML_F3W_WXIH0191_newtraining"
    # # modulename_model = "ML_F3W_WXIH0190_newtraining_4L_512N"
    # # modulename_model = "ML_F3W_WXIH0190"

    modulenames_used_for_training = ["ML_F3W_WXIH0190"]
    # modulenames_used_for_training = ["ML_F3W_WXIH0190", "ML_F3W_WXIH0191"]

    modulename_for_evaluation = "ML_F3W_WXIH0190"
    # modulename_for_evaluation = "ML_F3W_WXIH0191"
    # modulename_for_evaluation = "ML_F3W_WXIH0192"

    modelname = "regression_dnn"
    # modelname = "linreg"

    # nodes_per_layer = [128, 128, 64]
    nodes_per_layer = [512, 512, 512, 512, 64]

    # dropout_rate = 0.0
    dropout_rate = 0.05
    # dropout_rate = 0.1
    # dropout_rate = 0.2

    modeltag = ""
    # modeltag = "fraction0p1"

    override_full_model_name = False
    # new_model_name = "ML_F3W_WXIH0190_newtraining_4L_512N"
    # new_model_name = "ML_F3W_WXIH0190_ML_F3W_WXIH0191_newtraining"
    new_model_name = "ML_F3W_WXIH0190_newtraining"


    

    ### No need to change things below here -------------------------

    # --- Select device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    nch_per_erx = 37 if modulename_for_evaluation.startswith("ML") else 74
    nerx = 6 if modulename_for_evaluation.startswith("ML") else 12
    
    inputfolder = f"/eos/user/{os.getenv('USER')[0]}/{os.getenv('USER')}/hgcal/dnn_inputs/{modulename_for_evaluation}"

    # --- Load inputs and model ---
    inputs_train  = np.load(f"{inputfolder}/inputs_train.npy")
    inputs_val    = np.load(f"{inputfolder}/inputs_val.npy")
    targets_train = np.load(f"{inputfolder}/targets_train.npy")
    targets_val   = np.load(f"{inputfolder}/targets_val.npy")
    
    # --- Convert to tensors ---
    X_train = torch.tensor(inputs_train,  dtype=torch.float32).to(device)
    X_val   = torch.tensor(inputs_val,    dtype=torch.float32).to(device)
    y_train = torch.tensor(targets_train, dtype=torch.float32).squeeze().to(device)
    y_val   = torch.tensor(targets_val,   dtype=torch.float32).squeeze().to(device)

    # --- Load full chadc & split it by saved indices ---
    chadc_full  = np.load(f"{inputfolder}/chadc.npy").astype(int).squeeze()
    eventid_full  = np.load(f"{inputfolder}/eventid.npy").astype(int).squeeze()
    train_idx   = np.load(f"{inputfolder}/indices_train.npy")
    val_idx     = np.load(f"{inputfolder}/indices_val.npy")
    chadc_train = chadc_full[train_idx]
    chadc_val   = chadc_full[val_idx]
    eventid_train = eventid_full[train_idx]
    eventid_val   = eventid_full[val_idx]
    print("--> Loaded inputs and targets")

    # # --- Instantiate & load model ---
    # model = models.DNNFlex(input_dim=X_train.shape[1], nodes_per_layer=[128, 128, 64], dropout_rate=0.0, tag=modeltag).to(device)
    model = models.DNNFlex(input_dim=X_train.shape[1], nodes_per_layer=nodes_per_layer, dropout_rate=dropout_rate, tag=modeltag).to(device)
    if override_full_model_name:
        model.override_model_string(new_model_name)
    modelfolder = f"/eos/user/{os.getenv('USER')[0]}/{os.getenv('USER')}/hgcal/dnn_models/{'_'.join(modulenames_used_for_training)}/{model.get_model_string()}"
    plotfolder = f"plots/performance/{'_'.join(modulenames_used_for_training)}/{model.get_model_string()}/inputs_from_{modulename_for_evaluation}"
    os.makedirs(name=plotfolder, exist_ok=True)

    pth_to_load = f"{modelfolder}/{modelname}_best.pth"
    if not os.path.exists(pth_to_load):
        raise FileNotFoundError(f"Saved model '{modelfolder}/{modelname}_best.pth' not found.")
    model.load_state_dict(torch.load(pth_to_load, map_location=device))
    model.to(device)
    model.eval()
    print(f"--> Loaded model {model.get_model_string()}")

    npar_total, npar_trainable = utils.count_parameters(model=model)
    print(f"Total parameters:     {npar_total:,}")
    print(f"Trainable parameters: {npar_trainable:,}")

    # --- Predict separately for train and val ---
    with torch.no_grad():
        y_pred_train = model(X_train).squeeze().numpy()
        y_pred_val   = model(X_val).squeeze().numpy()
    print(f"--> Made predictions")

    # bring true y back to numpy
    y_train = y_train.squeeze().numpy()
    y_val   = y_val.squeeze().numpy()

    # --- Build the “combined” arrays for diagnostics ---
    inputs_combined    = np.concatenate([inputs_train, inputs_val], axis=0)
    y_true_combined    = np.concatenate([y_train,      y_val],      axis=0)
    y_pred_combined    = np.concatenate([y_pred_train, y_pred_val], axis=0)
    chadc_combined     = np.concatenate([chadc_train,  chadc_val],  axis=0)
    eventid_combined   = np.concatenate([eventid_train,  eventid_val],  axis=0)
    print(f"--> Concatenated train and val")



    # # --- Plot loss curves ---
    plot_loss(modelfolder=modelfolder, plotfolder=plotfolder)

    # Call for each set
    # plot_all_diagnostics(inputs=inputs_train, y_true=y_train, y_pred=y_pred_train, chadc=chadc_train, label_suffix="train_mean0", inputfolder=inputfolder, plotfolder=plotfolder, use_unstandardized=False)
    # plot_all_diagnostics(inputs=inputs_val, y_true=y_val, y_pred=y_pred_val, chadc=chadc_val, label_suffix="val_mean0", inputfolder=inputfolder, plotfolder=plotfolder, use_unstandardized=False)
    plot_all_diagnostics(inputs=inputs_combined, y_true=y_true_combined, y_pred=y_pred_combined, chadc=chadc_combined, label_suffix="combined_mean0", inputfolder=inputfolder, plotfolder=plotfolder, use_unstandardized=False)
    plot_coherent_noise(y_true=y_true_combined, y_pred=y_pred_combined, chadc=chadc_combined, eventid=eventid_combined, nch_per_erx=nch_per_erx, nerx=nerx, label_suffix="combined_mean0", inputfolder=inputfolder, plotfolder=plotfolder)
    # raise ValueError("stop")

    # Infer number of ADC channels from chadc
    n_channels = int(chadc_combined.max()) + 1
    n_samples = len(y_true_combined)
    assert n_samples % n_channels == 0, "Mismatch between samples and channels"

    n_events = n_samples // n_channels

    # Reshape predictions and truths
    y_combined_2d = y_true_combined.reshape((n_events, n_channels))
    y_pred_combined_2d = y_pred_combined.reshape((n_events, n_channels))
    residual_2d = y_combined_2d - y_pred_combined_2d

    # Compute per-channel RMS
    rms_true_per_channel = np.sqrt(np.mean(y_combined_2d**2, axis=0))
    rms_corrected_per_channel = np.sqrt(np.mean(residual_2d**2, axis=0))
    std_true_per_channel = np.std(y_combined_2d, axis=0)
    std_corrected_per_channel = np.std(residual_2d, axis=0)

    # Create a 1D histogram of input and corrected RMS per channel
    plt.figure(figsize=(8, 5))
    plt.hist(rms_true_per_channel, bins=30, alpha=0.6, label="Uncorrected", color='gray')
    plt.hist(rms_corrected_per_channel, bins=30, alpha=0.6, label="Corrected", color='tomato')
    plt.xlabel("RMS (ADC units)")
    plt.ylabel("Number of channels")
    plt.title("Per-channel RMS Before and After Correction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{plotfolder}/rms_comparison_per_channel.pdf")
    print(f"Saved per-channel RMS comparison: {plotfolder}/rms_comparison_per_channel.pdf")
    plt.close()

    # Create a 1D histogram of uncorrected and corrected standard deviation per channel
    plt.figure(figsize=(8, 5))
    plt.hist(std_true_per_channel, bins=30, alpha=0.6, label="Uncorrected", color='gray')
    plt.hist(std_corrected_per_channel, bins=30, alpha=0.6, label="Corrected", color='seagreen')
    plt.xlabel("Standard Deviation (ADC units)")
    plt.ylabel("Number of channels")
    plt.title("Per-channel Standard Deviation Before and After Correction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{plotfolder}/std_comparison_per_channel.pdf")
    print(f"Saved per-channel std comparison: {plotfolder}/std_comparison_per_channel.pdf")
    plt.close()

    # fractional improvement  (1 − corrected / uncorrected)
    frac_impr = 1.0 - (rms_corrected_per_channel / rms_true_per_channel)
    frac_impr_std = 1.0 - (std_corrected_per_channel / std_true_per_channel)
    plt.figure(figsize=(8, 5))
    plt.hist(frac_impr, bins=20, color="royalblue")
    plt.xlabel(r"Fractional improvement  $1-\mathrm{RMS}_{\rm corr}/\mathrm{RMS}_{\rm uncorr}$", fontsize=14)
    plt.ylabel("Number of channels", fontsize=14)
    plt.title("Per-channel fractional RMS improvement", fontsize=16)
    plt.axvline(frac_impr.mean(), color="k", ls="--", label=f"mean = {frac_impr.mean():.3%}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plotfolder}/rms_frac_improvement_per_channel.pdf")
    print(f"Saved: {plotfolder}/rms_frac_improvement_per_channel.pdf")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(frac_impr_std, bins=20, color="mediumseagreen")
    plt.xlabel(r"Fractional improvement  $1-\sigma_{\rm corr}/\sigma_{\rm uncorr}$", fontsize=14)
    plt.ylabel("Number of channels", fontsize=14)
    plt.title("Per-channel fractional standard deviation improvement", fontsize=16)
    plt.axvline(frac_impr_std.mean(), color="k", ls="--", label=f"mean = {frac_impr_std.mean():.3%}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plotfolder}/std_frac_improvement_per_channel.pdf")
    print(f"Saved: {plotfolder}/std_frac_improvement_per_channel.pdf")
    plt.close()

    # selected_channels = [0, 17, 19, 37, 74, 111, 148, 185]
    selected_channels = [0, 37]
    for ch in selected_channels:
        plotfolder_thischannel = f"{plotfolder}/channel{ch:03}"
        os.makedirs(plotfolder_thischannel, exist_ok=True)
        rms_uncorr = rms_true_per_channel[ch]
        rms_corr = rms_corrected_per_channel[ch]
        plt.figure(figsize=(8, 5))
        plt.hist(y_combined_2d[:, ch], bins=100, alpha=0.6, label=f"Uncorrected (RMS = {np.sqrt(np.mean(y_combined_2d[:, ch]**2)):.2f})", color="gray")
        plt.hist(residual_2d[:, ch], bins=100, alpha=0.6, label=f"Corrected (RMS = {np.sqrt(np.mean(residual_2d[:, ch]**2)):.2f})", color="tomato")
        # plt.yscale("log")
        plt.xlabel("ADC", fontsize=14)
        plt.ylabel("Entries", fontsize=14)
        plt.title(f"ADC Distribution in Channel {ch}", fontsize=16)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{plotfolder_thischannel}/adc_distribution_channel{ch:03}.pdf")
        print(f"Saved: {plotfolder_thischannel}/adc_distribution_channel{ch:03}.pdf")
        plt.close()

        for erx_idx in range(12):
            cm_col = inputs_combined[:, erx_idx].reshape((n_events, n_channels))[:, ch]
            label = f"CM e-Rx {erx_idx:02}"

            # True ADC vs. CM e-Rx
            utils.plot_y_vs_x_with_marginals(
                vals_x=cm_col,
                vals_y=y_combined_2d[:, ch],
                label_x=label,
                label_y=f"True ADC (Channel {ch})",
                label_profile="Mean ADC",
                output_filename=f"{plotfolder_thischannel}/2d_trueADC_vs_cmerx{erx_idx:02}_channel{ch:03}.pdf"
            )

            # Predicted ADC vs. CM e-Rx
            utils.plot_y_vs_x_with_marginals(
                vals_x=cm_col,
                vals_y=y_pred_combined_2d[:, ch],
                label_x=label,
                label_y=f"Predicted ADC (Channel {ch})",
                label_profile="Mean ADC",
                output_filename=f"{plotfolder_thischannel}/2d_predADC_vs_cmerx{erx_idx:02}_channel{ch:03}.pdf"
            )

            # Residual vs. CM e-Rx
            utils.plot_y_vs_x_with_marginals(
                vals_x=cm_col,
                vals_y=residual_2d[:, ch],
                label_x=label,
                label_y=f"Residual (Channel {ch})",
                label_profile="Mean residual",
                output_filename=f"{plotfolder_thischannel}/2d_residual_vs_cmerx{erx_idx:02}_channel{ch:03}.pdf"
            )

def plot_coherent_noise(y_true, y_pred, chadc, eventid, nch_per_erx, nerx, inputfolder, plotfolder, label_suffix=""):
    """
    Print direct / alternating sums, before and after correction,
    for every ERx board.

    Parameters
    ----------
    y_true, y_pred : (N,) float arrays
        True and DNN-predicted ADC values, same row order.
    chadc          : (N,) int array
        Channel ID for every row (0 … 37*nerx-1).
    nch_per_erx    : int
        Number of channels served by one ERx board (37 in your setup).
    nerx           : int
        Total number of ERx boards (6 or 12).
    """


    # # Undo standardization
    # targets_mean = np.load(f"{inputfolder}/targets_mean.npy")
    # y_true_unstandardized = y_true + targets_mean[chadc]
    # y_pred_unstandardized = y_pred + targets_mean[chadc]

    coh_true_list, inc_true_list, ratio_true_list = [], [], []
    coh_corr_list, inc_corr_list, ratio_corr_list = [], [], []
    all_dir_true_per_erx, all_alt_true_per_erx, all_dir_corr_per_erx, all_alt_corr_per_erx = [], [], [], []
    for erx in range(nerx):

        print(f"ERx {erx:2d}  ")
        order = np.lexsort((chadc, eventid))     # minor key first!

        chadc_ordered   = chadc[order]
        eventid_ordered = eventid[order]
        y_true_ordered  = y_true[order]
        y_pred_ordered  = y_pred[order]


        # ------- pick rows that belong to this ERx -----------------------
        mask  = (chadc_ordered >=  erx      * nch_per_erx) & (chadc_ordered <  (erx + 1) * nch_per_erx)
        idxs  = np.where(mask)[0]                 # integer positions

        adc_true = y_true_ordered[idxs]

        adc_pred = y_pred_ordered[idxs]
        adc_corr = adc_true - adc_pred

        # ── reshape to (n_events, 37) ────────────────────────────────────
        n_rows = adc_true.size
        assert n_rows % nch_per_erx == 0, "rows not multiple of nch_per_erx"
        n_evt  = n_rows // nch_per_erx

        adc_true_2d = adc_true.reshape(n_evt, nch_per_erx)
        adc_corr_2d = adc_corr.reshape(n_evt, nch_per_erx)

        # ── per-event direct & alternating sums ─────────────────────────
        dir_sums_true = adc_true_2d.sum(axis=1)                       # shape (n_evt,)
        alt_sums_true = adc_true_2d[:, ::2].sum(axis=1) - adc_true_2d[:, 1::2].sum(axis=1)

        dir_sums_corr = adc_corr_2d.sum(axis=1)
        alt_sums_corr = adc_corr_2d[:, ::2].sum(axis=1) - adc_corr_2d[:, 1::2].sum(axis=1)

        all_dir_true_per_erx.append(dir_sums_true)
        all_alt_true_per_erx.append(alt_sums_true)
        all_dir_corr_per_erx.append(dir_sums_corr)
        all_alt_corr_per_erx.append(alt_sums_corr)


        # print(f"var direct true: {dir_sums_true.var()}, direct corr: {dir_sums_corr.var()}")
        # print(f"var altern true: {alt_sums_true.var()}, altern corr: {alt_sums_corr.var()}")

        delta_true = dir_sums_true.var()-alt_sums_true.var()
        delta_corr = dir_sums_corr.var()-alt_sums_corr.var()

        inc_noise_true = np.std(alt_sums_true)/math.sqrt(nch_per_erx)
        coh_noise_true = np.sign(delta_true)*np.sqrt( np.abs(delta_true)) / nch_per_erx

        inc_noise_corr = np.std(alt_sums_corr)/math.sqrt(nch_per_erx)
        coh_noise_corr = np.sign(delta_corr)*np.sqrt( np.abs(delta_corr)) / nch_per_erx
        # print(f"direct, true -- std:{dir_sums_true.std()}")
        # print(f"direct, corr -- std:{dir_sums_corr.std()}")
        # print(f"altern, true -- std:{alt_sums_true.std()}")
        # print(f"altern, corr -- std:{alt_sums_corr.std()}")

        coh_true_list.append(coh_noise_true)
        inc_true_list.append(inc_noise_true)
        ratio_true_list.append(coh_noise_true / (coh_noise_true + inc_noise_true))
        coh_corr_list.append(coh_noise_corr)
        inc_corr_list.append(inc_noise_corr)
        ratio_corr_list.append(coh_noise_corr / (coh_noise_corr + inc_noise_corr))

        # draw the two sums
        overlay_hist(dir_sums_true,  alt_sums_true, "direct sum channels", "alternating sum channels", "Uncorrected per-event sums", outfilename=f"{os.path.join(plotfolder, 'sums_true')}_erx{erx:02d}.pdf")
        overlay_hist(dir_sums_corr,  alt_sums_corr, "direct sum channels (corrected)", "alternating sum channels (corrected)", "Corrected per-event sums", outfilename=f"{os.path.join(plotfolder, 'sums_corrected')}_erx{erx:02d}.pdf")


    # convert to arrays so indexing is easy
    coh_true_arr   = np.array(coh_true_list)
    inc_true_arr   = np.array(inc_true_list)
    ratio_true_arr = np.array(ratio_true_list)
    coh_corr_arr   = np.array(coh_corr_list)
    inc_corr_arr   = np.array(inc_corr_list)
    ratio_corr_arr = np.array(ratio_corr_list)
    erx_idx        = np.arange(nerx)

    # build common binning
    n_bins = 50
    all_dir_flat = np.concatenate(all_dir_true_per_erx)
    y_min, y_max = all_dir_flat.min(), all_dir_flat.max()
    bin_edges = np.linspace(y_min, y_max, n_bins+1)
    
    # fill matrix: rows = bins, cols = ERx
    hist2d = np.zeros((n_bins, nerx), dtype=int)
    for erx, vec in enumerate(all_dir_true_per_erx):
        hist2d[:, erx], _ = np.histogram(vec, bins=bin_edges)
    
    # plot
    fig, ax = plt.subplots(figsize=(7,5))
    extent = [-0.5, nerx-0.5, y_min, y_max]   # x from -0.5 to 5.5 etc.
    im = ax.imshow(hist2d, origin='lower', aspect='auto', extent=extent, cmap='viridis')
    ax.set_xlabel("ERx board index")
    ax.set_xticks(erx_idx)
    ax.set_ylabel("Direct sum (ADC)")
    ax.set_title("")
    cb = fig.colorbar(im, ax=ax); cb.set_label("Events")
    plt.tight_layout()
    plt.savefig(f"{plotfolder}/ds_true_vs_erx.pdf")

    
    # --- ratios corr / true -------------------------------------------------
    inc_ratio   = inc_corr_arr   / inc_true_arr
    coh_ratio   = coh_corr_arr   / coh_true_arr
    coh_inc_true = coh_true_arr / inc_true_arr
    coh_inc_corr = coh_corr_arr / inc_corr_arr

    # --- two-row figure with shared x-axis ---------------------------------
    fig = plt.figure(figsize=(7, 6))
    gs  = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.05)

    ax1 = fig.add_subplot(gs[0])   # main panel
    axr = fig.add_subplot(gs[1], sharex=ax1)   # ratio panel
    axc = fig.add_subplot(gs[2], sharex=ax1)  # coh / inc

    # ── main panel (exactly what you had) -----------------------------------
    ax1.plot(erx_idx, inc_true_arr,  'o-',  label='incoherent (true)', color='tab:blue')
    ax1.plot(erx_idx, coh_true_arr,  's-',  label='coherent  (true)',  color='tab:orange')
    ax1.plot(erx_idx, inc_corr_arr, 'o--', label='incoherent (corr)', color='tab:blue')
    ax1.plot(erx_idx, coh_corr_arr, 's--', label='coherent  (corr)',  color='tab:orange')
    
    # ── axis styling  (ticks inward, all four sides) -----------------------
    for ax in (ax1, axr, axc):
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, labelsize=12)

    ax1.set_ylabel('Noise (ADC)', fontsize=16, loc='top', labelpad=12)
    ax1.set_ylim(0., 3.)
    ax1.grid(ls='--', alpha=0.3)

    # combined legend
    h1,l1 = ax1.get_legend_handles_labels()
    ax1.legend(h1, l1, loc='upper right', fontsize=14)

    # ── ratio panel ---------------------------------------------------------
    axr.plot(erx_idx, inc_ratio,  'o--',  color='tab:blue')
    axr.plot(erx_idx, coh_ratio,  's--',  color='tab:orange')
    axr.set_xlabel('e-Rx', fontsize=16, loc='right', labelpad=8)
    axr.set_ylabel('corr / true', fontsize=14, loc='top', labelpad=12)
    axr.set_ylim(0., 1.1)          # adjust as you like
    axr.grid(ls='--', alpha=0.3)

    # ── ratio panel 2 : coh / inc --------------------------------------------
    axc.plot(erx_idx, coh_inc_true, 'D-',  color='black')
    axc.plot(erx_idx, coh_inc_corr, 'D--', color='black')
    axc.set_xlabel('e-Rx',  fontsize=16, loc='right', labelpad=8)
    axc.set_ylabel('coh / inc', fontsize=14, loc='top',  labelpad=8)
    axc.set_ylim(0., 2.)
    axc.grid(ls='--', alpha=0.3)

    # hide x-tick labels on the upper panels (shared axis)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(axr.get_xticklabels(), visible=False)

    plt.tight_layout()
    outfilename = os.path.join(plotfolder, 'noise_fractions_with_ratio.pdf')
    if label_suffix:
        outfilename = outfilename.replace(".pdf", f"_{label_suffix}.pdf")
    fig.savefig(os.path.join(plotfolder, 'noise_fractions_with_ratio.pdf'), bbox_inches='tight', pad_inches=0.05)



def overlay_hist(a, b, label_a, label_b, title, outfilename, bins=50):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    lo, hi  = min(a.min(), b.min()), max(a.max(), b.max())
    edges   = np.linspace(lo, hi, int(bins) + 1)
    plt.figure(figsize=(6,4))
    plt.hist(a, bins=edges, alpha=0.55, label=label_a)
    plt.hist(b, bins=edges, alpha=0.55, label=label_b)
    plt.title(title)
    plt.xlabel("sum (integrated ADC)")
    plt.ylabel("Event count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfilename)
    print(f"Wrote plot {outfilename}")
    plt.close()

def plot_loss(modelfolder, plotfolder):

    # --- Load losses ---
    if not os.path.exists(f"{modelfolder}/train_losses.npy") or not os.path.exists(f"{modelfolder}/val_losses.npy"):
        raise FileNotFoundError("Missing train_losses.npy or val_losses.npy.")

    train_losses = np.load(f"{modelfolder}/train_losses.npy")
    val_losses = np.load(f"{modelfolder}/val_losses.npy")

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    loss_plot_path = f"{plotfolder}/loss_curve.pdf"
    plt.savefig(loss_plot_path)
    print(f"Saved loss plot to: {loss_plot_path}")

def plot_all_diagnostics(inputs, y_true, y_pred, chadc, label_suffix, inputfolder, plotfolder, use_unstandardized=False):
    """Generate all diagnostic plots for a given dataset."""


    # Unstandardize inputs and/or targets if requested
    if use_unstandardized:
        # Load means and chadc
        inputs_mean = np.load(f"{inputfolder}/inputs_mean.npy")
        targets_mean = np.load(f"{inputfolder}/targets_mean.npy")
        print(targets_mean)
    
        # Rescale inputs
        inputs = inputs + inputs_mean
    
        # Rescale targets and predictions per-channel
        y_true = y_true + targets_mean[chadc]
        y_pred = y_pred + targets_mean[chadc]

    if y_true.shape != y_pred.shape:
        y_true = np.ravel(y_true)
        y_pred = np.ravel(y_pred)
    assert y_true.shape == y_pred.shape, "Shapes still do not match after flattening!"

    residual = y_true - y_pred
    rms_true = np.sqrt(np.mean(y_true**2))
    rms_corrected = np.sqrt(np.mean(residual**2))
    mean_true = np.mean(y_true)
    mean_corrected = np.mean(residual)

    print(f"[{label_suffix}] Mean True: {mean_true:.4f}, Mean Corrected: {mean_corrected:.4f}")
    print(f"[{label_suffix}] RMS True: {rms_true:.4f}, RMS Corrected: {rms_corrected:.4f}")

    # --- True vs Predicted
    utils.plot_y_vs_x_with_marginals(
        vals_x=y_true,
        vals_y=y_pred,
        label_x="True ADC",
        label_y="Predicted ADC",
        label_profile="Profile: mean predicted ADC",
        output_filename=f"{plotfolder}/true_vs_predicted_adc_{label_suffix}.pdf"
    )

    # --- Residual vs. each CM e-Rx
    for idx in range(12):
        utils.plot_y_vs_x_with_marginals(
            vals_x=inputs[:, idx],
            vals_y=residual,
            label_x=f"CM e-Rx {idx:02}",
            label_y="Corrected ADC (Residual)",
            label_profile="Profile: mean residual",
            output_filename=f"{plotfolder}/residual_vs_CMerx{idx:02}_{label_suffix}.pdf"
        )

    # --- Predicted vs. each CM e-Rx
    for idx in range(12):
        utils.plot_y_vs_x_with_marginals(
            vals_x=inputs[:, idx],
            vals_y=y_pred,
            label_x=f"CM e-Rx {idx:02}",
            label_y="Predicted ADC",
            label_profile="Profile: mean predicted ADC",
            output_filename=f"{plotfolder}/predicted_ADC_vs_CMerx{idx:02}_{label_suffix}.pdf"
        )

    # --- RMS Bar Plot
    plt.figure(figsize=(6, 5))
    plt.bar(["Uncorrected", "Corrected"], [rms_true, rms_corrected], color=["steelblue", "tomato"])
    plt.ylabel("RMS (ADC units)")
    plt.title(f"RMS Before and After Correction ({label_suffix})")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f"{plotfolder}/rms_comparison_{label_suffix}.pdf")
    print(f"Saved RMS comparison plot: {plotfolder}/rms_comparison_{label_suffix}.pdf")
    plt.close()



if __name__ == '__main__':
    main()
