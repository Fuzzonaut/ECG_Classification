"""Step 2: train the ECG1DCNN model on the cached data.

Usage:
    python train.py --data data/processed/data.npz
"""

import argparse
import json
import os

import torch.nn as nn
import torch.optim as optim

from ptbxl.data import load_npz
from ptbxl.model import ECG1DCNN
from ptbxl.training import get_device, make_datasets, make_loaders, plot_history, set_seed, train_model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/processed/data.npz")
    p.add_argument("--checkpoint", default="checkpoints/ecg_best_model.pt")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    splits, label_names, feature_names = load_npz(args.data)
    loaders = make_loaders(make_datasets(splits), batch_size=args.batch_size, seed=args.seed)

    num_features = splits["train"]["X"].shape[1]
    num_labels = splits["train"]["Z"].shape[1]
    model = ECG1DCNN(num_features=num_features, num_labels=num_labels)

    history = train_model(
        model,
        loaders,
        optimizer=optim.Adam(model.parameters(), lr=args.lr),
        criterion=nn.BCELoss(),
        epochs=args.epochs,
        device=get_device(),
        checkpoint_path=args.checkpoint,
        extra_checkpoint_info={
            "num_features": num_features,
            "num_labels": num_labels,
            "label_names": label_names,
            "feature_names": feature_names,
        },
    )

    with open(os.path.join(args.results_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    plot_history(history, save_path=os.path.join(args.results_dir, "training_curves.png"))
    print(f"Best checkpoint: {args.checkpoint}. Now run: python evaluate.py")


if __name__ == "__main__":
    main()
