"""Step 3: evaluate the selected checkpoint once on the held-out test fold.

Usage:
    python evaluate.py --data data/processed/data.npz --checkpoint checkpoints/ecg_best_model.pt
"""

import argparse
import json
import os

from ptbxl.data import load_npz
from ptbxl.metrics import compute_metrics, format_metrics, per_label_report
from ptbxl.training import get_device, load_model, make_datasets, make_loaders, predict


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/processed/data.npz")
    p.add_argument("--checkpoint", default="checkpoints/ecg_best_model.pt")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--split", default="test", choices=["valid", "test"])
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = get_device()

    splits, label_names, _ = load_npz(args.data)
    loader = make_loaders(make_datasets(splits))[args.split]
    model, ckpt = load_model(args.checkpoint, device)
    print(f"Loaded checkpoint from epoch {ckpt['epoch'] + 1} (val loss {ckpt['val_loss']:.4f})")

    y_true, y_prob = predict(model, loader, device)
    metrics = compute_metrics(y_true, y_prob, threshold=args.threshold)
    report = per_label_report(y_true, y_prob, label_names, threshold=args.threshold)

    print(f"\n=== {args.split} set ===")
    print(format_metrics(metrics))
    print("\nPer-label (top 15 by support):")
    print(report.head(15).round(3).to_string())

    with open(os.path.join(args.results_dir, f"metrics_{args.split}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    report.to_csv(os.path.join(args.results_dir, f"per_label_{args.split}.csv"))
    print(f"\nSaved results to {args.results_dir}/")


if __name__ == "__main__":
    main()
