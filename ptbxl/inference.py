"""Run the trained model on a new ECG.

    from ptbxl.inference import ECGPredictor

    predictor = ECGPredictor("model")
    signal = predictor.read_record("samples/clbbb")           # WFDB record (.hea + .dat)
    for r in predictor.predict(signal, age=67, sex="male")[:5]:
        print(r["code"], r["probability"], r["description"])

Not a medical device. For demonstration only.
"""

import json
import os

import numpy as np
import pandas as pd
import torch

from .data import build_tabular_features
from .model import ECG1DCNN

LEAD_NAMES = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
N_SAMPLES = 1000  # 10 s at 100 Hz

SEX_CODES = {"male": 0, "m": 0, "0": 0, "female": 1, "f": 1, "1": 1}  # PTB-XL: 0 = male, 1 = female


class ECGPredictor:
    def __init__(self, model_dir="model", device=None):
        self.device = device or torch.device("cpu")

        ckpt = torch.load(os.path.join(model_dir, "ecg_model.pt"), map_location=self.device)
        self.model = ECG1DCNN(ckpt["num_features"], ckpt["num_labels"])
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()
        self.label_names = list(ckpt["label_names"])

        with open(os.path.join(model_dir, "preprocessing.json")) as f:
            self.stats = json.load(f)
        with open(os.path.join(model_dir, "label_descriptions.json")) as f:
            self.label_info = json.load(f)

        self.feature_names = self.stats["feature_names"]
        self.scaler_mean = np.array(self.stats["scaler_mean"])
        self.scaler_scale = np.array(self.stats["scaler_scale"])
        self.lead_mean = np.array(self.stats["lead_mean"])
        self.lead_std = np.array(self.stats["lead_std"])

    # ------------------------------------------------------------------ #
    # Input
    # ------------------------------------------------------------------ #
    @staticmethod
    def read_record(path):
        """Read a WFDB record (path without extension) -> (1000, 12) array in mV at 100 Hz."""
        import wfdb

        signal, meta = wfdb.rdsamp(path)
        return ECGPredictor.prepare_signal(signal, meta["fs"], meta.get("sig_name"))

    @staticmethod
    def read_csv(path, fs=100):
        """Read a CSV with 12 columns (one per lead, in standard order) and one row per sample."""
        raw = pd.read_csv(path, header=None)
        names = None
        if pd.to_numeric(raw.iloc[0], errors="coerce").isna().any():  # first row is a header
            names, raw = raw.iloc[0].tolist(), raw.iloc[1:]
        return ECGPredictor.prepare_signal(raw.to_numpy(dtype=float), fs, names)

    @staticmethod
    def prepare_signal(signal, fs=100, lead_names=None):
        """Reorder leads to the standard order, resample to 100 Hz and take the first 10 s."""
        signal = np.asarray(signal, dtype=float)
        if signal.ndim != 2 or 12 not in signal.shape:
            raise ValueError(f"Expected a 12-lead signal, got shape {signal.shape}")
        if signal.shape[1] != 12:
            signal = signal.T

        if lead_names is not None:
            upper = [str(n).strip().upper() for n in lead_names]
            if sorted(upper) == sorted(LEAD_NAMES):
                signal = signal[:, [upper.index(n) for n in LEAD_NAMES]]

        if fs != 100:
            from scipy.signal import resample_poly

            signal = resample_poly(signal, 100, int(fs), axis=0)

        if signal.shape[0] < N_SAMPLES:
            raise ValueError(
                f"Need at least 10 s of signal ({N_SAMPLES} samples at 100 Hz), got {signal.shape[0]}"
            )
        signal = signal[:N_SAMPLES]
        return np.nan_to_num(signal)

    def encode_metadata(self, age=None, sex=None, height=None, weight=None,
                        infarction_stadium1=None, infarction_stadium2=None,
                        pacemaker=False, extra_beats=None):
        """Encode metadata exactly like the training features, then scale them."""
        if isinstance(sex, str):
            sex = SEX_CODES.get(sex.strip().lower())
        row = pd.DataFrame(
            {
                "age": [age],
                "sex": [sex],
                "height": [height],
                "weight": [weight],
                "infarction_stadium1": [infarction_stadium1],
                "infarction_stadium2": [infarction_stadium2],
                "pacemaker": ["ja, pacemaker" if pacemaker else None],
                "extra_beats": [extra_beats],
            }
        )
        for col in ("age", "sex", "height", "weight"):
            row[col] = pd.to_numeric(row[col], errors="coerce")

        X = build_tabular_features(row, median_height=self.stats["median_height"])
        X = X[self.feature_names].to_numpy(dtype=float)
        return (X - self.scaler_mean) / self.scaler_scale

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, signal, **metadata):
        """Return one dict per label, sorted by predicted probability (highest first)."""
        signal = (np.asarray(signal, dtype=float) - self.lead_mean) / self.lead_std
        ecg = torch.as_tensor(signal.T[None], dtype=torch.float32, device=self.device)
        tab = torch.as_tensor(self.encode_metadata(**metadata), dtype=torch.float32, device=self.device)

        probs = self.model(ecg, tab)[0].cpu().numpy()

        results = [
            {
                "code": code,
                "probability": float(p),
                "description": self.label_info.get(code, {}).get("description", ""),
                "superclass": self.label_info.get(code, {}).get("superclass", ""),
            }
            for code, p in zip(self.label_names, probs)
        ]
        return sorted(results, key=lambda r: r["probability"], reverse=True)
