#! /eos/user/a/areimers/torch-env/bin/python

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


import models
import utils


# -----------------------------
# Entry point
# -----------------------------
def main():
    # configure paths and layout
    cfg = EvalConfig(
        modulenames_used_for_training=["ML_F3W_WXIH0190"], 
        modulename_for_evaluation="ML_F3W_WXIH0190", 
        nodes_per_layer=[512, 512, 512, 512, 64],
        dropout_rate=0.0,
        modeltag="",
        inputfoldertag="", 
        ncmchannels=12
    )

    # load data
    io = DataIO(cfg=cfg)
    io.load_all()

    # DNN inference -> event x channel DataFrames same as measurements_df ---
    dnn_model = cfg.make_dnn_model(input_dim=io.get_split("train").inputs_flat.shape[1])
    model_dir = cfg.model_folder(model=dnn_model)
    weights_path = cfg.model_weights_path(model_folder=model_dir)
    dnn_inferencer = DNNInferencer(model=dnn_model, weights_path=weights_path, dtype=cfg.dnn_dtype, batch_size=cfg.dnn_batch_size)
    # Analytic inferencer
    analytic_inferencer = AnalyticInferencer(drop_constant_cm=True)    
    
    for split_name in ["combined"]:
        s = io.get_split(split_name)
        (variants, variants_with_cms) = build_variants(split=s, cfg=cfg, model_folder=model_dir, dnn_inferencer=dnn_inferencer, analytic_inferencer=analytic_inferencer, k_list=(0, 1, 3, 5, 10))
        (residuals, residuals_with_cms) = make_residuals(variants, cm_df=s.cm_df)  
        
        plot_split_diagnostics(split_name=split_name, cfg=cfg, model_folder=model_dir, cm_df=s.cm_df, variants=variants, variants_with_cms=variants_with_cms, residuals=residuals, residuals_with_cms=residuals_with_cms)
        

def plot_split_diagnostics(split_name: str, cfg: EvalConfig, model_folder: str, cm_df: pd.DataFrame, variants: Dict[str, pd.DataFrame], variants_with_cms: Dict[str, pd.DataFrame], residuals: Dict[str, pd.DataFrame], residuals_with_cms: Dict[str, pd.DataFrame]):
    # build plot folder consistent with your old layout
    train_tag = "_".join(cfg.modulenames_used_for_training)
    model_string = os.path.basename(model_folder)
    plot_dir = os.path.join("plots", "performance", f"{train_tag}{cfg.inputfoldertag}", model_string, f"inputs_from_{cfg.modulename_for_evaluation}", split_name)
    os.makedirs(plot_dir, exist_ok=True)
    

    # Cov and Corr plots for all variants and residuals
    plot_cov_corr(split_name=split_name, cfg=cfg, variants=variants_with_cms, residuals=residuals_with_cms, plot_dir=os.path.join(plot_dir, "covcorr"))

    plot_per_channel_diagnostics(split_name=split_name, variants=variants, residuals=residuals, plot_dir=os.path.join(plot_dir, "per_channel"))

    # 2d plots with marginals
    plot_vs_each_cm(split_name=split_name, cm_df=cm_df, variants=variants, residuals=residuals, plot_dir=os.path.join(plot_dir, "per_channel_2d_vs_cm"))

    # coh/inc noise ratios
    compute_and_plot_coherent_noise(split_name=split_name, cfg=cfg, variants=variants, residuals=residuals, plot_dir=os.path.join(plot_dir, "coherent_noise"), trunc_fracs=(1.0, 0.95, 0.90))


    # Eigen decompositions
    plot_all_eigenvalues(split_name=split_name, variants=variants, residuals=residuals, plot_dir=os.path.join(plot_dir, "eigenvalues"))
    plot_all_eigenvectors(cfg=cfg, split_name=split_name, variants=variants, residuals=residuals, k=4, plot_dir=os.path.join(plot_dir, "eigenvectors"))
    plot_all_projection_hists(split_name=split_name, variants=variants, residuals=residuals, k=4, plot_dir=os.path.join(plot_dir, "eigenprojections"))
    plot_loss(model_folder=model_folder, plot_dir=plot_dir)


# -----------------------------
# Configuration
# -----------------------------
@dataclass
class EvalConfig:
    # training/eval module selection
    modulenames_used_for_training: List[str]
    modulename_for_evaluation: str

    # path and layout knobs
    inputfoldertag: str = ""
    ncmchannels: int = 12
    nch_per_erx: Optional[int] = None
    nerx: Optional[int] = None

    # DNN identification
    username_load_model_from: str = "areimers"
    modelname: str = "regression_dnn"              # weight basename: {modelname}_best.pth
    nodes_per_layer: List[int] = None              # defaults set in infer_layout
    dropout_rate: float = 0.0
    modeltag: str = ""                             # tag recorded in DNNFlex string
    override_full_model_name: bool = False
    new_model_name: str = "ML_F3W_WXIH0190_newtraining"

    # inference knobs
    dnn_batch_size: int = 16384
    dnn_dtype: np.dtype = np.float32

    # derive defaults from module naming convention
    def infer_layout(self) -> None:
        if self.modulename_for_evaluation.startswith("ML"):
            self.nch_per_erx = 37 if self.nch_per_erx is None else self.nch_per_erx
            self.nerx = 6 if self.nerx is None else self.nerx
        else:
            self.nch_per_erx = 74 if self.nch_per_erx is None else self.nch_per_erx
            self.nerx = 12 if self.nerx is None else self.nerx

    # resolve EOS input folder
    def input_folder(self) -> str:
        user = os.getenv(key="USER", default="")
        if not user:
            raise RuntimeError("Environment variable USER not set")
        return f"/eos/user/{user[0]}/{user}/hgcal/dnn_inputs{self.inputfoldertag}/{self.modulename_for_evaluation}"

    # --- model folder resolution ---
    def make_dnn_model(self, input_dim: int) -> nn.Module:
        model = models.DNNFlex(input_dim=input_dim, nodes_per_layer=self.nodes_per_layer, dropout_rate=self.dropout_rate, tag=self.modeltag)
        if self.override_full_model_name: 
            model.override_model_string(self.new_model_name)
        return model

    def model_folder(self, model: nn.Module) -> str:
        user = self.username_load_model_from
        train_tag = "_".join(self.modulenames_used_for_training)
        return f"/eos/user/{user[0]}/{user}/hgcal/dnn_models{self.inputfoldertag}/{train_tag}/{model.get_model_string()}"

    def model_weights_path(self, model_folder: str) -> str:
        return os.path.join(model_folder, f"{self.modelname}_best.pth")

    def preds_cache_path(self, model_folder: str, split_name: str) -> str:
        return os.path.join(model_folder, f"predictions_{self.modulename_for_evaluation}_{split_name}.npy")


# -----------------------------
# Container for dataset splits
# -----------------------------
@dataclass
class SplitData:
    # flat arrays as loaded
    name: str
    targets_flat: np.ndarray
    inputs_flat: np.ndarray
    channels_flat: np.ndarray
    eventid_flat: np.ndarray

    # 2D matrices (rows sorted by eventid)
    measurements_df: pd.DataFrame
    measurements_with_cms_df: pd.DataFrame
    inputs_df: pd.DataFrame
    cm_df: pd.DataFrame

    # metadata
    event_ids: np.ndarray
    n_channels: int


# -----------------------------
# Input/output handling
# -----------------------------
class DataIO:
    def __init__(self, cfg: EvalConfig):
        # store configuration and infer layout
        self.cfg = cfg
        self.cfg.infer_layout()
        self._loaded = False

    # load arrays and build train/val/combined splits
    def load_all(self) -> None:
        folder = self.cfg.input_folder()

        # load arrays
        inputs_train = np.load(file=os.path.join(folder, "inputs_train.npy"), mmap_mode="r")
        inputs_val   = np.load(file=os.path.join(folder, "inputs_val.npy"),   mmap_mode="r")
        targets_train = np.load(file=os.path.join(folder, "targets_train.npy"), mmap_mode="r")
        targets_val   = np.load(file=os.path.join(folder, "targets_val.npy"),   mmap_mode="r")
        chadc_full  = np.load(file=os.path.join(folder, "chadc.npy"),   mmap_mode="r").astype(int).squeeze()
        eventid_full = np.load(file=os.path.join(folder, "eventid.npy"), mmap_mode="r").astype(int).squeeze()
        train_idx   = np.load(file=os.path.join(folder, "indices_train.npy"), mmap_mode="r")
        val_idx     = np.load(file=os.path.join(folder, "indices_val.npy"),   mmap_mode="r")
        with open(os.path.join(folder, "colnames.json")) as f:
            colnames = json.load(f)

        # split indices to train/val
        chadc_train, chadc_val       = chadc_full[train_idx],   chadc_full[val_idx]
        eventid_train, eventid_val   = eventid_full[train_idx], eventid_full[val_idx]

        # build train split
        self.train = self._build_split(name="train", targets_flat=np.asarray(a=targets_train).squeeze(), inputs_flat=np.asarray(a=inputs_train), channels_flat=np.asarray(a=chadc_train), eventid_flat=np.asarray(a=eventid_train), colnames_inputs=colnames)

        # build val split
        self.val = self._build_split(name="val", targets_flat=np.asarray(a=targets_val).squeeze(), inputs_flat=np.asarray(a=inputs_val), channels_flat=np.asarray(a=chadc_val), eventid_flat=np.asarray(a=eventid_val), colnames_inputs=colnames)

        # build combined split from concatenated flats (re-pivoted by event id)
        targets_combined  = np.concatenate([np.asarray(a=targets_train).squeeze(), np.asarray(a=targets_val).squeeze()])
        inputs_combined   = np.concatenate([np.asarray(a=inputs_train), np.asarray(a=inputs_val)], axis=0)
        channels_combined = np.concatenate([np.asarray(a=chadc_train),  np.asarray(a=chadc_val)])
        eventid_combined  = np.concatenate([np.asarray(a=eventid_train), np.asarray(a=eventid_val)])

        self.combined = self._build_split(name="combined", targets_flat=targets_combined, inputs_flat=inputs_combined, channels_flat=channels_combined, eventid_flat=eventid_combined, colnames_inputs=colnames)

        # mark loaded
        self._loaded = True

    # get a split by name
    def get_split(self, name: str) -> SplitData:
        if not self._loaded:
            raise RuntimeError("Call load_all() first")
        if not hasattr(self, name):
            raise ValueError(f"Unknown split: {name}")
        return getattr(self, name)

    # build one split: pivot to event×channel and extract per-event CM table
    def _build_split(self, name: str, targets_flat: np.ndarray, inputs_flat: np.ndarray, channels_flat: np.ndarray, eventid_flat: np.ndarray, colnames_inputs: List[str]) -> SplitData:
        if len(colnames_inputs) != inputs_flat.shape[1]:
            raise ValueError(f"colnames.json has {len(colnames_inputs)} names but inputs have {inputs_flat.shape[1]} columns.")

        # measurement matrix, rows sorted by eventid
        meas_df = self._pivot_measurements(values=targets_flat, channels=channels_flat, eventid=eventid_flat)

        # inputs + cm matrices, also sorted by eventid internally
        inputs_df, cm_df = self._build_input_and_cm_df(inputs_flat=inputs_flat, eventid_flat=eventid_flat, ncm=self.cfg.ncmchannels, colnames_inputs=colnames_inputs)

        meas_with_cm_df = add_cms_to_measurements_df(measurements_df=meas_df, cm_df=cm_df, drop_constant_cm=False)

        # sanity: indices must match exactly (same order)
        if not np.array_equal(meas_df.index.values, inputs_df.index.values):
            raise RuntimeError("Row index mismatch between measurements_df and inputs_df.")

        n_channels = meas_df.shape[1]
        event_ids_sorted = meas_df.index.to_numpy()

        return SplitData(name=name, targets_flat=targets_flat, inputs_flat=inputs_flat, channels_flat=channels_flat, eventid_flat=eventid_flat, measurements_df=meas_df, inputs_df=inputs_df, cm_df=cm_df, measurements_with_cms_df=meas_with_cm_df, event_ids=event_ids_sorted, n_channels=n_channels)

    # reshape measurements into event × channel DataFrame (rows sorted by eventid)
    @staticmethod
    def _pivot_measurements(values: np.ndarray, channels: np.ndarray, eventid: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame({"value": values, "channel": channels.astype(int), "eventid": eventid.astype(int)})
        wide = df.pivot(index="eventid", columns="channel", values="value")
        wide.columns = [f"ch_{c:03d}" for c in wide.columns]
        return wide.sort_index().reindex(columns=sorted(wide.columns))

    # build inputs_df (all features) and cm_df (CM-only subset), both row-sorted by eventid
    @staticmethod
    def _build_input_and_cm_df(inputs_flat: np.ndarray, eventid_flat: np.ndarray, ncm: int, colnames_inputs: List[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        if inputs_flat.shape[1] < ncm:
            raise ValueError(f"Requested ncm={ncm} but inputs have only {inputs_flat.shape[1]} columns")

        # unique event IDs with first occurrence positions, then sort by eventid
        unique_ids, first_pos = np.unique(eventid_flat, return_index=True)
        order = np.argsort(unique_ids)
        event_ids_sorted = unique_ids[order]
        rows = first_pos[order]

        # full inputs df in original column order recorded in colnames_inputs
        inputs_df = pd.DataFrame(inputs_flat[rows, :], index=event_ids_sorted, columns=colnames_inputs)

        # CM subset (preserve the original order from colnames_inputs)
        cm_cols = [c for c in colnames_inputs if c.startswith("cm_erx")]
        if len(cm_cols) != ncm:
            raise ValueError(f"Found {len(cm_cols)} CM columns by name ({cm_cols[:5]}...), but cfg.ncmchannels={ncm}.")
        cm_df = inputs_df[cm_cols]

        return (inputs_df, cm_df)


class DNNInferencer:
    def __init__(self, model: nn.Module, weights_path: str, dtype: np.dtype, batch_size: int):
        self.model = model
        self.weights_path = weights_path
        self.dtype = dtype
        self.batch_size = batch_size
        self.device = torch.device("cpu")
        if not os.path.exists(self.weights_path): 
            raise FileNotFoundError(f"Saved model '{self.weights_path}' not found.")
        state = torch.load(self.weights_path, map_location="cpu")
        self.model.load_state_dict(state_dict=state)
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=self.dtype)
        preds = []
        with torch.no_grad():
            for i in range(0, X.shape[0], self.batch_size):
                batch = torch.from_numpy(X[i:i+self.batch_size]).to(self.device)
                out = self.model(batch).squeeze()
                preds.append(out.detach().cpu().numpy())
        return np.concatenate(preds, axis=0)


def pivot_flat_preds_to_event_channel(preds_flat: np.ndarray, eventid_flat: np.ndarray, channels_flat: np.ndarray, reference_meas_df: pd.DataFrame) -> pd.DataFrame:
    df_long = pd.DataFrame({"value": preds_flat, "eventid": eventid_flat.astype(int), "channel": channels_flat.astype(int)})
    df_long["eventid"] = pd.Categorical(df_long["eventid"], categories=reference_meas_df.index.values, ordered=True)
    wide = df_long.pivot(index="eventid", columns="channel", values="value")
    wide.columns = [f"ch_{c:03d}" for c in wide.columns]
    wide = wide.reindex(columns=reference_meas_df.columns)      # exact same channels, same order
    preds_df = wide.reindex(index=reference_meas_df.index)       # exact same event order
    if not np.array_equal(preds_df.index.values, reference_meas_df.index.values):
        raise RuntimeError("DNN preds: event index mismatch.")
    if not np.array_equal(preds_df.columns.values, reference_meas_df.columns.values):
        raise RuntimeError("DNN preds: channel columns mismatch.")
    return preds_df


def run_dnn_for_split(split: SplitData, cfg: EvalConfig, model_folder: str, inferencer: DNNInferencer) -> pd.DataFrame:
    cache_path = cfg.preds_cache_path(model_folder=model_folder, split_name=split.name)
    if os.path.exists(cache_path):
        preds_flat = np.load(cache_path, mmap_mode="r")
        print(f"--> Loaded existing DNN predictions from {cache_path}")
    else:
        preds_flat = inferencer(split.inputs_flat)
        np.save(cache_path, preds_flat)
        print(f"--> Made new DNN predictions and saved them to {cache_path}")
    if preds_flat.shape[0] != split.targets_flat.shape[0]: 
        raise RuntimeError(f"DNN returned {preds_flat.shape[0]} preds but split has {split.targets_flat.shape[0]} samples.")
    return pivot_flat_preds_to_event_channel(preds_flat=preds_flat, eventid_flat=split.eventid_flat, channels_flat=split.channels_flat, reference_meas_df=split.measurements_df)


class AnalyticInferencer:
    def __init__(self, drop_constant_cm: bool = True):
        self.drop_constant_cm = drop_constant_cm
        self._cache = {}  # key: id(split) -> {"W": np.ndarray, "keep_mask": np.ndarray}

    def _compute_weights(self, meas_df: pd.DataFrame, cm_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if not np.array_equal(meas_df.index.values, cm_df.index.values):
            raise RuntimeError("AnalyticInferencer: meas_df and cm_df must share identical row indices.")
        # drop strictly-constant CM columns (no epsilon)
        cm_std = cm_df.std(axis=0)
        keep_mask = (cm_std > 0).values if self.drop_constant_cm else np.ones(cm_df.shape[1], dtype=bool)
        if not keep_mask.any():
            raise RuntimeError("AnalyticInferencer: all CM columns are constant—cannot invert Σ_cc.")

        C = cm_df.loc[:, cm_df.columns[keep_mask]].to_numpy()
        M = meas_df.to_numpy()
        E = M.shape[0]

        Sigma_mc = (M.T @ C) / E                                                  # (N, K')
        Sigma_cc = (C.T @ C) / E                                                  # (K', K')
        W = Sigma_mc @ np.linalg.inv(Sigma_cc)                                    # (N, K')
        return W, keep_mask

    def fit(self, split: SplitData) -> None:
        W, keep_mask = self._compute_weights(meas_df=split.measurements_df, cm_df=split.cm_df)
        self._cache[id(split)] = {"W": W, "keep_mask": keep_mask, "index": split.measurements_df.index.values, "columns": split.measurements_df.columns.values}

    def predict(self, split: SplitData) -> pd.DataFrame:
        if id(split) not in self._cache: 
            self.fit(split)

        entry = self._cache[id(split)]
        W, keep_mask = entry["W"], entry["keep_mask"]
        C = split.cm_df.loc[:, split.cm_df.columns[keep_mask]].to_numpy()
        Y = C @ W.T
        preds = pd.DataFrame(Y, index=split.measurements_df.index, columns=split.measurements_df.columns)

        # sanity (exact alignment)
        if not np.array_equal(preds.index.values, entry["index"]): 
            raise RuntimeError("AnalyticInferencer: event index changed since fit().")
        if not np.array_equal(preds.columns.values, entry["columns"]): 
            raise RuntimeError("AnalyticInferencer: channel columns changed since fit().")
        return preds

    def predict_k(self, split: SplitData, k: int) -> pd.DataFrame:
        base = self.predict(split=split)
        if k <= 0: return base

        # residuals (E×N)
        Rm = (split.measurements_df - base).to_numpy()

        # channel covariance
        Cr = (Rm.T @ Rm) / Rm.shape[0]

        # eigendecomposition (descending)
        vals, vecs = np.linalg.eigh(Cr)
        order = np.argsort(vals)[::-1]
        vals = vals[order]; vecs = vecs[:, order]

        k = int(min(k, vecs.shape[1] - 1))
        if k == 0: 
            return base

        U = vecs[:, :k]
        lam = vals[:k]
        R0 = Cr - (U * lam) @ U.T
        V  = np.linalg.solve(R0, U)
        D  = U.T @ V + np.diag(1.0 / lam)
        Ahat = Rm @ V @ np.linalg.inv(D)
        
        corr = Ahat @ U.T
        corrected = base + pd.DataFrame(corr, index=base.index, columns=base.columns)
        return corrected

    

def add_cms_to_measurements_df(measurements_df: pd.DataFrame, cm_df: pd.DataFrame, drop_constant_cm: bool = True) -> pd.DataFrame:
    X = pd.concat([measurements_df, cm_df], axis=1)
    if drop_constant_cm and cm_df.shape[1] > 0:
        # drop CM columns with zero variance (avoid NaNs in correlation)
        cm_std = cm_df.std(axis=0)
        keep = cm_std[cm_std > 0].index
        dropped = [c for c in cm_df.columns if c not in keep]
        if dropped:
            print(f"[info] Dropping {len(dropped)} constant CM columns: {dropped}")
        X = pd.concat([measurements_df, cm_df[keep]], axis=1)
    return X

def build_variants(split: SplitData, cfg: EvalConfig, model_folder: str, dnn_inferencer: Optional[DNNInferencer] = None, analytic_inferencer: Optional[AnalyticInferencer] = None, k_list: Tuple[int, ...] = (0, 1, 3)) -> tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Return a dict of aligned event x channel DataFrames for this split.
    Keys may include: 'true', 'dnn', 'wiener_k=0, 'wiener_k=1', 'wiener_k=3', ...
    """
    variants: Dict[str, pd.DataFrame] = {}
    variants_with_cms: Dict[str, pd.DataFrame] = {}

    # 1) always include ground truth
    variants["true"] = split.measurements_df
    variants_with_cms["true"] = add_cms_to_measurements_df(measurements_df=split.measurements_df, cm_df=split.cm_df, drop_constant_cm=False)

    # 2) DNN (cached through run_dnn_for_split)
    if dnn_inferencer is not None:
        variants["dnn"] = run_dnn_for_split(split=split, cfg=cfg, model_folder=model_folder, inferencer=dnn_inferencer)
        variants_with_cms["dnn"] = add_cms_to_measurements_df(measurements_df=variants["dnn"], cm_df=split.cm_df, drop_constant_cm=False)

    # 3) Analytic (Wiener and Wiener+K)
    if analytic_inferencer is not None:

        # Extra K modes (k=0 is standard Wiener)
        for k in k_list:
            if k < 0: 
                continue
            df_k = analytic_inferencer.predict_k(split=split, k=k)
            variants[f"wiener_k={k}"] = df_k
            variants_with_cms[f"wiener_k={k}"] = add_cms_to_measurements_df(measurements_df=variants[f"wiener_k={k}"], cm_df=split.cm_df, drop_constant_cm=False)

    # 4) final sanity: exact same index/columns as truth
    for key, df in variants.items():
        if not np.array_equal(df.index.values, variants["true"].index.values): 
            raise RuntimeError(f"Variant '{key}' has mismatched event index.")
        if not np.array_equal(df.columns.values, variants["true"].columns.values): 
            raise RuntimeError(f"Variant '{key}' has mismatched channel columns.")
    for key, df in variants_with_cms.items():
        if not np.array_equal(df.index.values, variants_with_cms["true"].index.values): 
            raise RuntimeError(f"Variant '{key}' has mismatched event index.")
        if not np.array_equal(df.columns.values, variants_with_cms["true"].columns.values): 
            raise RuntimeError(f"Variant '{key}' has mismatched channel columns.")

    return variants, variants_with_cms

# ---------- Stats utilities: covariance, correlation, residuals ----------

def compute_cov(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise empirical covariance with NaN-aware averaging (events x channels)."""
    # mask of valid entries (NaN = missing)
    mask = df.notna().astype(float)
    X = df.fillna(0.0).to_numpy()
    M = mask.to_numpy()

    # per-pair counts and sums
    N = M.T @ M                    # valid-event counts per channel pair
    S = X.T @ X                    # sum of products (zeros where NaN)

    # average only over valid events
    with np.errstate(invalid="ignore", divide="ignore"):
        C = S / N
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

    return pd.DataFrame(C, index=df.columns, columns=df.columns)


def corr_from_cov(cov: pd.DataFrame) -> pd.DataFrame:
    """Convert covariance to correlation (safe when diagonal has zeros)."""
    d = np.diag(cov.to_numpy())
    inv = np.zeros_like(d, dtype=float)
    pos = d > 0
    inv[pos] = 1.0 / np.sqrt(d[pos])
    D = np.diag(inv)
    R = D @ cov.to_numpy() @ D
    # numerical guard
    np.clip(R, -1.0, 1.0, out=R)
    return pd.DataFrame(R, index=cov.index, columns=cov.columns)


def compute_cov_corr(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper returning (cov, corr)."""
    cov = compute_cov(df)
    corr = corr_from_cov(cov)
    return (cov, corr)


def make_residuals(variants: Dict[str, pd.DataFrame], cm_df: pd.DataFrame) -> tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Build residual matrices for all prediction variants:
    residuals[key] = true - variants[key]  (for all keys != 'true')
    """
    if "true" not in variants:
        raise RuntimeError("make_residuals: 'true' matrix missing from variants.")
    ref = variants["true"]

    residuals: Dict[str, pd.DataFrame] = {}
    residuals_with_cms: Dict[str, pd.DataFrame] = {}
    for key, df in variants.items():
        if key == "true": 
            continue
        if not df.index.equals(ref.index) or not df.columns.equals(ref.columns):
            raise RuntimeError(f"make_residuals: alignment mismatch for '{key}'.")
        residuals[key] = ref - df
        residuals_with_cms[key] = add_cms_to_measurements_df(measurements_df=residuals[key], cm_df=cm_df, drop_constant_cm=False)
    return (residuals, residuals_with_cms)


def plot_cov_corr(split_name: str, cfg: EvalConfig, variants: Dict[str, pd.DataFrame], residuals: Dict[str, pd.DataFrame], plot_dir: str):
    os.makedirs(plot_dir, exist_ok=True)
    
    # --- Cov/Correlation for truth and predictions
    for key, df in variants.items():
        cov_true, corr_true = compute_cov_corr(df)
        utils.plot_covariance(df=cov_true, nch_per_erx=cfg.nch_per_erx, title=f"Covariance ({key}, {split_name})", xtitle="channel i", ytitle="channel j", ztitle="cov(i,j)", zrange=(-4., 4.), output_filename=os.path.join(plot_dir, f"Covariance_{key}_{split_name}.pdf"))
        utils.plot_covariance(df=corr_true, nch_per_erx=cfg.nch_per_erx, title=f"Correlation ({key}, {split_name})", xtitle="channel i", ytitle="channel j", ztitle="corr(i,j)", zrange=(-1., 1.), output_filename=os.path.join(plot_dir, f"Correlation_{key}_{split_name}.pdf"))

    # --- Cov/Correlation for residuals of each predictor
    for key, res_df in residuals.items():
        cov_res, corr_res = compute_cov_corr(res_df)
        utils.plot_covariance(df=cov_res, nch_per_erx=cfg.nch_per_erx, title=f"Covariance (residuals: {key}, {split_name})", xtitle="channel i", ytitle="channel j", ztitle="cov(i,j)", zrange=(-4., 4.), output_filename=os.path.join(plot_dir, f"Covariance_residuals_{key}_{split_name}.pdf"))
        utils.plot_covariance(df=corr_res, nch_per_erx=cfg.nch_per_erx, title=f"Correlation (residuals: {key}, {split_name})", xtitle="channel i", ytitle="channel j", ztitle="corr(i,j)", zrange=(-1., 1.), output_filename=os.path.join(plot_dir, f"Correlation_residuals_{key}_{split_name}.pdf"))

def compute_per_channel_stats(true_df: pd.DataFrame, pred_df: pd.DataFrame, res_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Return per-channel RMS/STD for true and corrected (pred)."""
    if not true_df.index.equals(pred_df.index) or not true_df.columns.equals(pred_df.columns):
        raise RuntimeError("compute_per_channel_stats: alignment mismatch.")

    true = true_df.to_numpy()
    res  = res_df.to_numpy()

    rms_true = np.sqrt((true ** 2).mean(axis=0))
    rms_corr = np.sqrt((res  ** 2).mean(axis=0))
    rms_frac_improvement = 1.0 - (rms_corr / rms_true)

    std_true = true.std(axis=0, ddof=0)
    std_corr = res.std(axis=0, ddof=0)
    std_frac_improvement = 1.0 - (std_corr / std_true)

    return {"rms_true": rms_true, "rms_corr": rms_corr, "rms_frac_improvement": rms_frac_improvement, "std_true": std_true, "std_corr": std_corr, "std_frac_improvement": std_frac_improvement}

def plot_per_channel_diagnostics(split_name: str, variants: Dict[str, pd.DataFrame], residuals: Dict[str, pd.DataFrame], plot_dir: str) -> None:
    """Per-channel RMS/STD histograms + fractional improvement for each predictor."""
    os.makedirs(plot_dir, exist_ok=True)

    # color palette consistent with your previous script
    method_color = {"dnn": "tomato", "wiener_k=0": "seagreen", "wiener_k=1": "orange", "wiener_k=3": "pink"}

    for key, pred in variants.items():
        if key == "true":
            continue

        stats = compute_per_channel_stats(true_df=variants["true"], pred_df=pred, res_df=residuals[key])

        # ---- per-method plots (own subfolder)
        col = method_color.get(key, "royalblue")

        utils.plot_hist_overlay_pair(a=stats["rms_true"], b=stats["rms_corr"], bins=30, label_a=f"Uncorrected (mean: {stats['rms_true'].mean():.3})", label_b=f"{key} corrected (mean: {stats['rms_corr'].mean():.3})", color_a="gray", color_b=col, xlabel="RMS (ADC units)", title=f"Per-channel RMS — {key} ({split_name})", outpath=os.path.join(plot_dir, f"rms_comparison_per_channel_{key}_{split_name}.pdf"))
        utils.plot_hist_overlay_pair(a=stats["std_true"], b=stats["std_corr"], bins=30, label_a=f"Uncorrected (mean: {stats['std_true'].mean():.3})", label_b=f"{key} corrected (mean: {stats['std_corr'].mean():.3})", color_a="gray", color_b=col, xlabel="Standard Deviation (ADC units)", title=f"Per-channel STD — {key} ({split_name})", outpath=os.path.join(plot_dir, f"std_comparison_per_channel_{key}_{split_name}.pdf"))

        # Fractional improvement hist
        utils.plot_hist_single(x=stats["rms_frac_improvement"], bins=80, color=col, xlabel=r"Fractional improvement  $1-\mathrm{RMS}_{\rm corr}/\mathrm{RMS}_{\rm uncorr}$", title=f"Per-channel fractional RMS improvement — {key} ({split_name})", outpath=os.path.join(plot_dir, f"rms_frac_improvement_per_channel_{key}.pdf"), show_mean_line=True)
        utils.plot_hist_single(x=stats["std_frac_improvement"], bins=80, color=col, xlabel=r"Fractional improvement  $1-\mathrm{STD}_{\rm corr}/\mathrm{STD}_{\rm uncorr}$", title=f"Per-channel fractional RMS improvement — {key} ({split_name})", outpath=os.path.join(plot_dir, f"std_frac_improvement_per_channel_{key}.pdf"), show_mean_line=True)

def plot_vs_each_cm(split_name: str, cm_df: pd.DataFrame, variants: Dict[str, pd.DataFrame], residuals: Dict[str, pd.DataFrame], plot_dir: str) -> None:
    os.makedirs(plot_dir, exist_ok=True)

    # sanity
    if not cm_df.index.equals(variants["true"].index):
        raise RuntimeError("CM df and measurements index mismatch.")

    n_channels = variants["true"].shape[1]

    # measurements, flattened event-major
    y_true_flat = variants["true"].to_numpy().ravel()

    for cm_name in cm_df.columns:
        subdir = os.path.join(plot_dir, cm_name)
        os.makedirs(subdir, exist_ok=True)

        # repeat the per-event CM value once per channel so it aligns with flattened matrices
        x_flat = np.repeat(cm_df[cm_name].to_numpy(), n_channels)

        # measurements vs CM
        utils.plot_y_vs_x_with_marginals(vals_x=x_flat, vals_y=y_true_flat, label_x=f"{cm_name} (ADC)", label_y="Uncorrected (ADC)", label_profile="profile", output_filename=os.path.join(subdir, f"uncorr_vs_{cm_name}_{split_name}.pdf"), nbins_x=80, nbins_y=80)

        # predictions and residuals vs CM
        for key, df_pred in variants.items():
            if key == "true": 
                continue
            y_pred_flat = df_pred.to_numpy().ravel()
            y_res_flat  = residuals[key].to_numpy().ravel()

            utils.plot_y_vs_x_with_marginals(vals_x=x_flat, vals_y=y_pred_flat, label_x=f"{cm_name} (ADC)", label_y=f"{key} corrected (ADC)", label_profile="profile", output_filename=os.path.join(subdir, f"{key}_corr_vs_{cm_name}_{split_name}.pdf"), nbins_x=80, nbins_y=80)

            utils.plot_y_vs_x_with_marginals(vals_x=x_flat, vals_y=y_res_flat,  label_x=f"{cm_name} (ADC)", label_y=f"{key} residual (ADC)",  label_profile="profile",  output_filename=os.path.join(subdir, f"{key}_residual_vs_{cm_name}_{split_name}.pdf"), nbins_x=80, nbins_y=80)

# ---------- Coherent / Incoherent noise: computations ----------
@dataclass
class CoherentNoiseResult:
    method: str
    trunc_frac: float
    erx_idx: np.ndarray
    coh_true: np.ndarray
    inc_true: np.ndarray
    coh_corr: np.ndarray
    inc_corr: np.ndarray
    coh_ratio: np.ndarray
    inc_ratio: np.ndarray
    coh_over_inc_true: np.ndarray
    coh_over_inc_corr: np.ndarray

def erx_channel_blocks(n_channels: int, nch_per_erx: int, nerx: int) -> List[np.ndarray]:
    if n_channels != nch_per_erx * nerx: 
        raise RuntimeError(f"Channel count {n_channels} != {nch_per_erx}*{nerx}.")
    return [np.arange(erx*nch_per_erx, (erx+1)*nch_per_erx, dtype=int) for erx in range(nerx)]

def dir_alt_sums_per_erx(block_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    direct = block_2d.sum(axis=1)
    alternating = block_2d[:, ::2].sum(axis=1) - block_2d[:, 1::2].sum(axis=1)
    return direct, alternating

def coh_inc_from_sums(direct: np.ndarray, alternating: np.ndarray, nch_per_erx: int, trunc_frac: float = 1.0) -> Tuple[float, float]:
    rms_dir = utils.truncated_rms(direct, trunc_frac)
    rms_alt = utils.truncated_rms(alternating, trunc_frac)
    delta = rms_dir**2 - rms_alt**2
    inc = rms_alt / np.sqrt(nch_per_erx)
    coh = np.sign(delta) * np.sqrt(abs(delta)) / nch_per_erx
    return coh, inc

def compute_coherent_noise_for_method(method: str, true_df: pd.DataFrame, res_df: pd.DataFrame, nch_per_erx: int, nerx: int, trunc_frac: float = 1.0) -> CoherentNoiseResult:
    n_channels = true_df.shape[1]
    blocks = erx_channel_blocks(n_channels=n_channels, nch_per_erx=nch_per_erx, nerx=nerx)

    coh_true, inc_true, coh_corr, inc_corr = [], [], [], []
    for cols in blocks:
        true_2d = true_df.iloc[:, cols].to_numpy()
        corr_2d = res_df.iloc[:,  cols].to_numpy()
        dir_true, alt_true = dir_alt_sums_per_erx(true_2d)
        dir_corr, alt_corr = dir_alt_sums_per_erx(corr_2d)

        c_t, i_t = coh_inc_from_sums(direct=dir_true, alternating=alt_true, nch_per_erx=nch_per_erx, trunc_frac=trunc_frac)
        c_c, i_c = coh_inc_from_sums(direct=dir_corr, alternating=alt_corr, nch_per_erx=nch_per_erx, trunc_frac=trunc_frac)
        coh_true.append(c_t)
        inc_true.append(i_t)
        coh_corr.append(c_c)
        inc_corr.append(i_c)

    coh_true = np.asarray(coh_true)
    inc_true = np.asarray(inc_true)
    coh_corr = np.asarray(coh_corr)
    inc_corr = np.asarray(inc_corr)

    with np.errstate(divide="ignore", invalid="ignore"):
        inc_ratio = np.nan_to_num(inc_corr / inc_true, nan=0.0)
        coh_ratio = np.nan_to_num(coh_corr / coh_true, nan=0.0)
        coh_over_inc_true = np.nan_to_num(coh_true / inc_true, nan=0.0)
        coh_over_inc_corr = np.nan_to_num(coh_corr / inc_corr, nan=0.0)

    return CoherentNoiseResult(method=method, trunc_frac=trunc_frac, erx_idx=np.arange(nerx), coh_true=coh_true, inc_true=inc_true, coh_corr=coh_corr, inc_corr=inc_corr, coh_ratio=coh_ratio, inc_ratio=inc_ratio, coh_over_inc_true=coh_over_inc_true, coh_over_inc_corr=coh_over_inc_corr)

def compute_coherent_noise_all_methods(cfg: EvalConfig, variants: Dict[str, pd.DataFrame], residuals: Dict[str, pd.DataFrame], trunc_frac: float = 1.0) -> Dict[str, CoherentNoiseResult]:
    true_df = variants["true"]
    out: Dict[str, CoherentNoiseResult] = {}
    for method, res_df in residuals.items():
        out[method] = compute_coherent_noise_for_method(method=method, true_df=true_df, res_df=res_df, nch_per_erx=cfg.nch_per_erx, nerx=cfg.nerx, trunc_frac=trunc_frac)
    return out


# ---------- Coherent / Incoherent noise: plotting only ----------

def plot_coherent_noise_from_result(split_name: str, result: CoherentNoiseResult, plot_dir: str) -> None:
    os.makedirs(plot_dir, exist_ok=True)

    fig = plt.figure(figsize=(7, 6))
    gs  = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.05)
    ax1 = fig.add_subplot(gs[0]); axr = fig.add_subplot(gs[1], sharex=ax1); axc = fig.add_subplot(gs[2], sharex=ax1)

    ax1.plot(result.erx_idx, result.inc_true, "o-",  label="incoherent (raw)",   color="tab:blue")
    ax1.plot(result.erx_idx, result.coh_true, "s-",  label="coherent (raw)",     color="tab:orange")
    ax1.plot(result.erx_idx, result.inc_corr, "o--", label="incoherent (corr.)", color="tab:blue")
    ax1.plot(result.erx_idx, result.coh_corr, "s--", label="coherent (corr.)",   color="tab:orange")

    for ax in (ax1, axr, axc):
        ax.tick_params(axis="both", direction="in", top=True, bottom=True, left=True, right=True, labelsize=12)
        ax.grid(ls="--", alpha=0.3)

    ax1.set_ylabel("Noise (ADC)", fontsize=16, loc="top", labelpad=12)
    ax1.set_ylim(0., max(ax1.get_ylim()[1], 3.))
    ax1.legend(loc="upper right", fontsize=12)

    axr.plot(result.erx_idx, result.inc_ratio, "o--", color="tab:blue")
    axr.plot(result.erx_idx, result.coh_ratio, "s--", color="tab:orange")
    axr.set_ylabel("corr./raw", fontsize=14, loc="top", labelpad=10)
    axr.set_ylim(0., 1.1)

    axc.plot(result.erx_idx, result.coh_over_inc_true, "D-",  color="black")
    axc.plot(result.erx_idx, result.coh_over_inc_corr, "D--", color="black")
    axc.set_xlabel("e-Rx", fontsize=16, loc="right", labelpad=8)
    axc.set_ylabel("coh/inc", fontsize=14, loc="top", labelpad=8)
    axc.set_ylim(0., max(axc.get_ylim()[1], 2.))

    plt.setp(ax1.get_xticklabels(), visible=False); plt.setp(axr.get_xticklabels(), visible=False)
    plt.tight_layout()

    frac_tag = f"{int(round(result.trunc_frac * 100))}"
    outname = f"noise_fractions_with_ratio_{result.method}_{split_name}_trunc-{frac_tag}.pdf"
    fig.savefig(os.path.join(plot_dir, outname), bbox_inches="tight", pad_inches=0.05)
    plt.close()



def compute_coherent_noise_all_methods_multi(cfg: EvalConfig, variants: Dict[str, pd.DataFrame], residuals: Dict[str, pd.DataFrame], trunc_fracs: Tuple[float, ...]) -> Dict[str, List[CoherentNoiseResult]]:
    out: Dict[str, List[CoherentNoiseResult]] = {}
    for method in residuals.keys():
        out[method] = [compute_coherent_noise_for_method(method=method, true_df=variants["true"], res_df=residuals[method], nch_per_erx=cfg.nch_per_erx, nerx=cfg.nerx, trunc_frac=f) for f in trunc_fracs]
    return out

def compute_and_plot_coherent_noise(split_name: str, cfg: EvalConfig, variants: Dict[str, pd.DataFrame], residuals: Dict[str, pd.DataFrame], plot_dir: str, trunc_fracs: Tuple[float, ...] = (1.0,)) -> Dict[str, List[CoherentNoiseResult]]:
    os.makedirs(plot_dir, exist_ok=True)
    results_by_method = compute_coherent_noise_all_methods_multi(cfg=cfg, variants=variants, residuals=residuals, trunc_fracs=trunc_fracs)
    for method, results_list in results_by_method.items():
        for res in results_list:
            subdir = os.path.join(plot_dir, method)
            plot_coherent_noise_from_result(split_name=split_name, result=res, plot_dir=subdir)
    return results_by_method

# Eigendecomposition
@dataclass
class EigDecompResult:
    method: str
    eigenvalues: np.ndarray         # shape (N,)
    eigenvectors: np.ndarray        # shape (N, N) columns = eigenvectors, descending λ
    channel_order: np.ndarray       # integer positions of columns in df (0..N-1)

def compute_eigendecomp_for_method(method: str, df: pd.DataFrame) -> EigDecompResult:
    # channel covariance across events (N x N); reuse our NaN-safe compute_cov
    cov = compute_cov(df)
    # symmetric -> eigh; sort descending
    vals, vecs = np.linalg.eigh(cov.to_numpy())
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    return EigDecompResult(method=method, eigenvalues=vals, eigenvectors=vecs, channel_order=np.arange(cov.shape[0]))

def plot_eigen_spectrum(split_name: str, eig: EigDecompResult, output_filename: str) -> None:
    x = np.arange(1, eig.eigenvalues.size + 1)

    # log-y (clip at tiny positive to avoid -inf)
    vals = np.clip(eig.eigenvalues, 1e-12, None)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(x, vals, marker='o', lw=1)
    ax.set_xlabel("mode index", loc="right")
    ax.set_ylabel("eigenvalue", loc="top")
    ax.grid(ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_filename)
    plt.close(fig)

def plot_all_eigenvalues(split_name: str, variants: Dict[str, pd.DataFrame], residuals: Dict[str, pd.DataFrame], plot_dir: str) -> None:
    # Eigen decompositions
    os.makedirs(plot_dir, exist_ok=True)
    for method, variant in variants.items():
        eig = compute_eigendecomp_for_method(method=method, df=variant)
        plot_eigen_spectrum(split_name=split_name, eig=eig, output_filename=os.path.join(plot_dir, f"eigenvalues_{eig.method}_{split_name}.pdf"))
    for method, variant in residuals.items():
        eig = compute_eigendecomp_for_method(method=method, df=variant)
        plot_eigen_spectrum(split_name=split_name, eig=eig, output_filename=os.path.join(plot_dir, f"eigenvalues_residuals_{eig.method}_{split_name}.pdf"))


def reshape_eigenvector_erx(v: np.ndarray, nch_per_erx: int, nerx: int) -> np.ndarray:
    # v is length N = nch_per_erx * nerx; reshape to (nerx, nch_per_erx) for a block heatmap
    if v.size != nch_per_erx * nerx:
        raise RuntimeError(f"Eigenvector length {v.size} != {nch_per_erx}*{nerx}.")
    # layout: rows = ERx index (0..nerx-1), columns = channel within ERx (0..nch_per_erx-1)
    return v.reshape(nerx, nch_per_erx)

def plot_topk_eigenvectors_1d(eig: EigDecompResult, output_filename: str, k: int, nch_per_erx: int, nerx: int) -> None:
    k = int(min(k, eig.eigenvectors.shape[1]))

    # (a) line plot across channel index
    fig, ax = plt.subplots(figsize=(6.4, 3.4))

    for i in range(k):
        v = eig.eigenvectors[:, i]
        lam = eig.eigenvalues[i]
        plt.plot(np.arange(v.size), v, label=f'Mode {i+1} ($\lambda$={lam:.3g})')
    
    for pos in range(0, nch_per_erx*(nerx+1), nch_per_erx):
        plt.axvline(pos, color='black', linestyle='--', linewidth=1)
    
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.ylim((-0.3, 0.3))
    plt.xlabel('Channel')
    plt.ylabel('Eigenvector component')
    plt.legend(ncol=2, fontsize='small')
    fig.tight_layout()
    fig.savefig(output_filename)
    plt.close(fig)

        
def plot_topk_eigenvectors_2d(eig: EigDecompResult, output_filename: str, k: int, nch_per_erx: int, nerx: int):
    k = int(min(k, eig.eigenvectors.shape[1]))
    for i in range(k):
        v = eig.eigenvectors[:, i]
        lam = eig.eigenvalues[i]

        # (b) ERx heatmap (nerx x nch_per_erx)
        mat = reshape_eigenvector_erx(v, nch_per_erx=nch_per_erx, nerx=nerx)
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        im = ax.imshow(mat, aspect="auto", cmap="coolwarm", interpolation="nearest")
        ax.set_xlabel("channel in ERx", loc="right")
        ax.set_ylabel("ERx index", loc="top")
        ax.set_title(f"mode {i+1}, $\lambda$={lam:.3g}")
        cb = fig.colorbar(im, ax=ax); cb.set_label("component")
        fig.tight_layout()
        fig.savefig(output_filename.replace(eig.method, f"{eig.method}_mode{i+1}"))
        plt.close(fig)



def plot_all_eigenvectors(cfg: EvalConfig, split_name: str, variants: Dict[str, pd.DataFrame], residuals: Dict[str, pd.DataFrame], k: int , plot_dir: str) -> None:
    
    plotdir_1d = plot_dir + "_1d"
    plotdir_2d = plot_dir + "_2d"
    os.makedirs(plotdir_1d, exist_ok=True)
    os.makedirs(plotdir_2d, exist_ok=True)

    for method, variant in variants.items():
        eig = compute_eigendecomp_for_method(method=method, df=variant)
        plot_topk_eigenvectors_1d(eig=eig, output_filename=os.path.join(plotdir_1d, f"eigenvectors_{eig.method}_{split_name}.pdf"), k=k, nch_per_erx=cfg.nch_per_erx, nerx=cfg.nerx)
        plot_topk_eigenvectors_2d(eig=eig, output_filename=os.path.join(plotdir_2d, f"eigenvectors_2d_{eig.method}_{split_name}.pdf"), k=k, nch_per_erx=cfg.nch_per_erx, nerx=cfg.nerx)
    for method, variant in residuals.items():
        eig = compute_eigendecomp_for_method(method=method, df=variant)
        plot_topk_eigenvectors_1d(eig=eig, output_filename=os.path.join(plotdir_1d, f"eigenvectors_residuals_{eig.method}_{split_name}.pdf"), k=k, nch_per_erx=cfg.nch_per_erx, nerx=cfg.nerx)
        plot_topk_eigenvectors_2d(eig=eig, output_filename=os.path.join(plotdir_2d, f"eigenvectors_residuals_2d_{eig.method}_{split_name}.pdf"), k=k, nch_per_erx=cfg.nch_per_erx, nerx=cfg.nerx)


def compute_projections_onto_topk(df: pd.DataFrame, eig: EigDecompResult, k: int) -> List[np.ndarray]:
    # residual matrix R: E x N ; columns align with eigenvectors columns
    R = df.to_numpy()  # E x N
    V = eig.eigenvectors[:, :k]  # N x k  (orthonormal from eigh)
    # projections: E x k
    P = R @ V
    return [P[:, i] for i in range(k)]

def plot_projection_hists(df: pd.DataFrame, eig: EigDecompResult, k: int, output_filename: str) -> None:
    projections = compute_projections_onto_topk(df=df, eig=eig, k=k)

    bins = np.histogram_bin_edges(np.concatenate(projections), bins=25)
    fig, ax = plt.subplots()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, p in enumerate(projections):
        mu  = np.mean(p)
        var = np.var(p)
        sig = np.sqrt(var)
        color = colors[i % len(colors)]
        ax.hist(p, bins=bins, density=True, linewidth=1.6, histtype="step", label=f"$v_{i}$: $\mu$={mu:.1f}, $\sigma^2$={var:.1f}", color=color)
        if sig > 0:
            x = np.linspace(bins[0], bins[-1], 600)
            pdf = (1.0 / (np.sqrt(2*np.pi)*sig)) * np.exp(-0.5*((x - mu)/sig)**2)
            ax.plot(x, pdf, linestyle="--", linewidth=1.4, color=color, label=f"v{i} Gaussian")
    ax.set_xlabel(f"Projection onto Eigenvector $v_i$")
    ax.set_ylabel("Event faction / bin")
    ax.legend(ncol=1)
    fig.tight_layout()
    fig.savefig(output_filename)
    plt.close(fig)

def plot_all_projection_hists(split_name: str, variants: Dict[str, pd.DataFrame], residuals: Dict[str, pd.DataFrame], k: int, plot_dir: str):
    os.makedirs(plot_dir, exist_ok=True)
    for method, variant in variants.items():
        eig = compute_eigendecomp_for_method(method=method, df=variant)
        plot_projection_hists(df=variant, eig=eig, k=k, output_filename=os.path.join(plot_dir, f"eigenprojections_{eig.method}_{split_name}.pdf"))
    for method, variant in residuals.items():
        eig = compute_eigendecomp_for_method(method=method, df=variant)
        plot_projection_hists(df=variant, eig=eig, k=k, output_filename=os.path.join(plot_dir, f"eigenprojections_residuals_{eig.method}_{split_name}.pdf"))



def plot_loss(model_folder, plot_dir):

    # --- Load losses ---
    if not os.path.exists(f"{model_folder}/train_losses.npy") or not os.path.exists(f"{model_folder}/val_losses.npy"):
        raise FileNotFoundError("Missing train_losses.npy or val_losses.npy.")

    train_losses = np.load(f"{model_folder}/train_losses.npy")
    val_losses = np.load(f"{model_folder}/val_losses.npy")

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    loss_plot_path = f"{plot_dir}/loss_curve.pdf"
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Saved loss plot to: {loss_plot_path}")


if __name__ == "__main__":
    main()