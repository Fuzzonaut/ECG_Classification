"""Step 4 (optional): package the trained model for sharing.

Writes:
    model/ecg_model.pt              weights + dimensions + label names (no optimizer state)
    model/preprocessing.json        scaler / normalisation statistics from training
    model/label_descriptions.json   human-readable meaning of each SCP code
    samples/                        a few test-fold ECGs (WFDB) + their metadata, for the demo

Usage:
    python export_model.py --data-dir data/ptb-xl
"""

import argparse
import json
import os
import shutil

import pandas as pd
import torch

from ptbxl.data import build_labels, load_metadata

# One clean example per diagnosis: test fold only, a single diagnostic label.
SAMPLE_CODES = ["NORM", "CLBBB", "CRBBB", "IMI", "ASMI", "LAFB", "LVH", "1AVB"]
SUPERCLASS_NAMES = {
    "NORM": "Normal ECG",
    "MI": "Myocardial infarction",
    "STTC": "ST/T change",
    "CD": "Conduction disturbance",
    "HYP": "Hypertrophy",
}


def _clean(v):
    """NaN -> None and numpy scalars -> Python, so the value is JSON-serialisable."""
    if pd.isna(v):
        return None
    return v.item() if hasattr(v, "item") else v


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/ptb-xl")
    p.add_argument("--checkpoint", default="checkpoints/ecg_best_model.pt")
    p.add_argument("--stats", default="data/processed/preprocessing.json")
    p.add_argument("--model-dir", default="model")
    p.add_argument("--samples-dir", default="samples")
    args = p.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.samples_dir, exist_ok=True)

    # ---- weights ----
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    slim = {k: ckpt[k] for k in ("model_state_dict", "num_features", "num_labels", "label_names", "feature_names")}
    slim["trained_epochs"] = ckpt["epoch"] + 1
    torch.save(slim, os.path.join(args.model_dir, "ecg_model.pt"))
    shutil.copy(args.stats, os.path.join(args.model_dir, "preprocessing.json"))

    # ---- label descriptions ----
    df, scp = load_metadata(args.data_dir)
    info = {
        code: {
            "description": str(scp.loc[code, "description"]),
            "superclass": SUPERCLASS_NAMES.get(str(scp.loc[code, "diagnostic_class"]), ""),
        }
        for code in ckpt["label_names"]
    }
    with open(os.path.join(args.model_dir, "label_descriptions.json"), "w") as f:
        json.dump(info, f, indent=2)

    # ---- demo samples ----
    Z, label_names = build_labels(df, scp)
    test = (df.strat_fold == 10).to_numpy()
    samples = []
    for code in SAMPLE_CODES:
        j = label_names.index(code)
        candidates = [
            i for i in range(len(df))
            if test[i] and Z[i].sum() == 1 and Z[i, j] == 1 and 18 <= (df.age.iloc[i] or 0) <= 89
        ]
        if not candidates:
            print(f"No clean sample found for {code}, skipping")
            continue
        i = candidates[0]
        row = df.iloc[i]
        src = os.path.join(args.data_dir, row.filename_lr)
        for ext in (".hea", ".dat"):
            shutil.copy(src + ext, args.samples_dir)
        samples.append(
            {
                "name": f"{code}: {info[code]['description']}",
                "record": os.path.basename(row.filename_lr),
                "ecg_id": int(df.index[i]),
                "true_labels": [code],
                "age": _clean(row.age),
                "sex": "female" if row.sex == 1 else "male",
                "height": _clean(row.height),
                "weight": _clean(row.weight),
                "infarction_stadium1": _clean(row.infarction_stadium1),
                "infarction_stadium2": _clean(row.infarction_stadium2),
                "pacemaker": row.pacemaker == "ja, pacemaker",
                "extra_beats": _clean(row.extra_beats),
            }
        )
    with open(os.path.join(args.samples_dir, "samples.json"), "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Model exported to {args.model_dir}/, {len(samples)} sample ECGs in {args.samples_dir}/")


if __name__ == "__main__":
    main()
