"""Loading and preprocessing PTB-XL.

This module is a direct port of the data-preparation part of the original
notebook. The preprocessing choices (features, imputation, encodings, split)
are kept as they were; only clear bugs were fixed (see README, "Changes from
the original notebook").
"""

import ast
import os

import numpy as np
import pandas as pd
import wfdb
from sklearn.preprocessing import StandardScaler

# Order of the tabular features fed into the model.
FEATURE_NAMES = [
    "age",
    "sex",
    "height",
    "weight",
    "infarction_stadium1",
    "infarction_stadium2",
    "pacemaker",
    "extra_beats",
]


# --------------------------------------------------------------------------- #
# Raw loading
# --------------------------------------------------------------------------- #
def load_metadata(data_dir):
    """Load `ptbxl_database.csv` and the *diagnostic* rows of `scp_statements.csv`."""
    df = pd.read_csv(os.path.join(data_dir, "ptbxl_database.csv"), index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    df.patient_id = df.patient_id.astype(int)
    df.nurse = df.nurse.astype("Int64")
    df.site = df.site.astype("Int64")
    df.validated_by = df.validated_by.astype("Int64")

    scp = pd.read_csv(os.path.join(data_dir, "scp_statements.csv"), index_col=0)
    scp = scp[scp.diagnostic == 1]
    return df, scp


def load_raw_data(df, sampling_rate, data_dir):
    """Read every WFDB record listed in `df`. Returns an array of shape (N, T, 12)."""
    files = df.filename_lr if sampling_rate == 100 else df.filename_hr
    signals = [wfdb.rdsamp(os.path.join(data_dir, f))[0] for f in files]
    return np.array(signals)


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #
def build_labels(df, scp):
    """Multi-hot target matrix over all diagnostic SCP statements.

    A record is positive for a statement if that statement appears in its
    `scp_codes` (regardless of the likelihood value), same as the original.

    Returns
    -------
    Z : np.ndarray, shape (N, n_labels), int
    label_names : list[str]  (sorted alphabetically)
    """
    label_names = sorted(scp.index)
    col = {name: j for j, name in enumerate(label_names)}

    Z = np.zeros((len(df), len(label_names)), dtype=int)
    for i, codes in enumerate(df.scp_codes):
        for code in codes:
            if code in col:
                Z[i, col[code]] = 1
    return Z, label_names


# --------------------------------------------------------------------------- #
# Tabular features
# --------------------------------------------------------------------------- #
def median_valid_height(df):
    """Median of plausible heights (>= 50 cm), used to impute missing/invalid ones."""
    return float(df["height"].where(df["height"] >= 50).median())


def build_tabular_features(df, median_height=None):
    """Hand-crafted metadata features, as designed in the original notebook.

    `median_height` defaults to the median valid height of `df` itself (training).
    At inference time pass the value saved from training instead.
    """
    X = pd.DataFrame(index=df.index)

    X["age"] = df.age.fillna(0)
    X["sex"] = df.sex.astype(float).fillna(0)

    # Height: values < 50 cm are treated as invalid and, like missing values,
    # replaced by the median of the valid heights.
    # (The original notebook computed this but never added the column to X.)
    if median_height is None:
        median_height = median_valid_height(df)
    X["height"] = df["height"].apply(lambda x: median_height if pd.isna(x) or x < 50 else x)

    X["weight"] = df.weight.fillna(0)

    X["infarction_stadium1"] = (
        df.infarction_stadium1.map(
            {
                "unknown": 0,
                "Stadium I": 1,
                "Stadium I-II": 2,
                "Stadium II": 3,
                "Stadium II-III": 4,
                "Stadium III": 5,
            }
        )
        .fillna(0)
        .astype(float)
    )

    X["infarction_stadium2"] = (
        df.infarction_stadium2.map(
            {"unknown": 0, "Stadium I": 1, "Stadium II": 2, "Stadium III": 3}
        )
        .fillna(0)
        .astype(float)
    )

    X["pacemaker"] = (df.pacemaker == "ja, pacemaker").astype(float)

    # Anything that is not exactly one of the three known strings becomes 0.
    extra_beats_map = {"SVES": 0.5, "VES": 1.0, "SVES+VES": 1.5}
    X["extra_beats"] = df.extra_beats.map(lambda v: extra_beats_map.get(v, 0.0)).astype(float)

    return X[FEATURE_NAMES]


# --------------------------------------------------------------------------- #
# Split + scaling
# --------------------------------------------------------------------------- #
def split_by_fold(df, X, signals, Z):
    """Use the official patient-wise stratified folds: 1-8 train, 9 valid, 10 test."""
    assert X.shape[0] == signals.shape[0] == Z.shape[0] == df.shape[0]

    masks = {
        "train": (df.strat_fold <= 8).to_numpy(),
        "valid": (df.strat_fold == 9).to_numpy(),
        "test": (df.strat_fold == 10).to_numpy(),
    }
    return {
        split: {"X": X[m], "Y": signals[m], "Z": Z[m]}
        for split, m in masks.items()
    }


def scale_tabular(splits):
    """Fit a StandardScaler on train tabular features and apply it to every split."""
    scaler = StandardScaler().fit(splits["train"]["X"])
    for s in splits.values():
        s["X"] = scaler.transform(s["X"])
    return scaler


def normalize_signals(splits):
    """Per-lead z-normalisation using statistics from the training set.

    Signals have shape (N, T, 12), so the per-lead statistics are taken over
    axes (0, 1). (The original notebook used axes (0, 2), which produced
    per-*timestep* statistics of shape (T,) instead of the intended (12,).)
    """
    train = splits["train"]["Y"]
    mean_per_lead = train.mean(axis=(0, 1))  # (12,)
    std_per_lead = train.std(axis=(0, 1))  # (12,)
    for s in splits.values():
        s["Y"] = (s["Y"] - mean_per_lead) / std_per_lead
    return mean_per_lead, std_per_lead


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
def save_npz(path, splits, label_names, feature_names=FEATURE_NAMES):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arrays = {}
    for split, d in splits.items():
        arrays[f"X_{split}"] = np.asarray(d["X"], dtype="float32")
        arrays[f"Y_{split}"] = np.asarray(d["Y"], dtype="float32")
        arrays[f"Z_{split}"] = np.asarray(d["Z"], dtype="float32")
    arrays["label_names"] = np.array(label_names)
    arrays["feature_names"] = np.array(feature_names)
    np.savez(path, **arrays)


def load_npz(path):
    """Load the cached arrays. Returns (splits, label_names, feature_names)."""
    data = np.load(path)
    splits = {
        split: {k: data[f"{k}_{split}"] for k in ("X", "Y", "Z")}
        for split in ("train", "valid", "test")
    }
    label_names = [str(x) for x in data["label_names"]]
    feature_names = [str(x) for x in data["feature_names"]]
    return splits, label_names, feature_names
