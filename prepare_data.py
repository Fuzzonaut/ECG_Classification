"""Step 1: read raw PTB-XL, build features/labels, split, normalise, cache to .npz.

Usage:
    python prepare_data.py --data-dir data/ptb-xl --out data/processed/data.npz
"""

import argparse
import json
import os

from ptbxl.data import (
    build_labels,
    build_tabular_features,
    load_metadata,
    load_raw_data,
    median_valid_height,
    normalize_signals,
    save_npz,
    scale_tabular,
    split_by_fold,
)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/ptb-xl",
                   help="folder containing ptbxl_database.csv, scp_statements.csv, records100/")
    p.add_argument("--out", default="data/processed/data.npz")
    p.add_argument("--sampling-rate", type=int, default=100, choices=[100, 500])
    args = p.parse_args()

    print("Loading metadata...")
    df, scp = load_metadata(args.data_dir)

    print(f"Loading {len(df)} ECG records at {args.sampling_rate} Hz (takes a few minutes)...")
    signals = load_raw_data(df, args.sampling_rate, args.data_dir)

    Z, label_names = build_labels(df, scp)
    X = build_tabular_features(df)
    print(f"Signals {signals.shape}, tabular {X.shape}, labels {Z.shape}")
    print(f"Records with >1 diagnostic label: {(Z.sum(axis=1) > 1).sum()}")

    splits = split_by_fold(df, X, signals, Z)
    scaler = scale_tabular(splits)
    lead_mean, lead_std = normalize_signals(splits)

    for name, d in splits.items():
        print(f"{name:<6} X={d['X'].shape} Y={d['Y'].shape} Z={d['Z'].shape}")

    save_npz(args.out, splits, label_names, list(X.columns))
    print(f"Saved to {args.out}")

    # Everything needed to preprocess a *new* ECG exactly like the training data.
    stats = {
        "sampling_rate": args.sampling_rate,
        "feature_names": list(X.columns),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "lead_mean": lead_mean.tolist(),
        "lead_std": lead_std.tolist(),
        "median_height": median_valid_height(df),
    }
    stats_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), "preprocessing.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved preprocessing statistics to {stats_path}")


if __name__ == "__main__":
    main()
