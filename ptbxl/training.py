"""Training / inference helpers.

Training setup is the same as the original notebook (BCELoss, Adam lr=1e-3,
batch size 64, 10 epochs, checkpoint on best validation loss). What changed:

* The test set is no longer evaluated every epoch. It is only touched once,
  after training, with the selected checkpoint.
* Epoch loss is averaged per sample instead of per batch.
* Validation is monitored with macro AUROC / macro F1 instead of label-wise
  accuracy (see `ptbxl.metrics`).
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .metrics import compute_metrics
from .model import ECG1DCNN


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_datasets(splits):
    """TensorDatasets of (ecg (N,T,12), tabular, labels) for each split."""
    def t(a):
        return torch.as_tensor(np.asarray(a), dtype=torch.float32)

    return {
        name: TensorDataset(t(d["Y"]), t(d["X"]), t(d["Z"]))
        for name, d in splits.items()
    }


def make_loaders(datasets, batch_size=64, seed=42):
    g = torch.Generator().manual_seed(seed)
    return {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, generator=g),
        "valid": DataLoader(datasets["valid"], batch_size=batch_size),
        "test": DataLoader(datasets["test"], batch_size=batch_size),
    }


def _to_model_input(ecg, tabular, device):
    # Stored as (N, T, 12); Conv1d expects (N, 12, T).
    return ecg.permute(0, 2, 1).to(device), tabular.to(device)


@torch.no_grad()
def predict(model, loader, device):
    """Return (y_true, y_prob) as numpy arrays for a whole loader."""
    model.eval()
    probs, targets = [], []
    for ecg, tabular, labels in loader:
        ecg, tabular = _to_model_input(ecg, tabular, device)
        probs.append(model(ecg, tabular).cpu().numpy())
        targets.append(labels.numpy())
    return np.concatenate(targets), np.concatenate(probs)


def train_model(model, loaders, optimizer, criterion, epochs=10, device=None,
                checkpoint_path="checkpoints/ecg_best_model.pt", extra_checkpoint_info=None):
    """Train, keep the checkpoint with the lowest validation loss, return history."""
    device = device or get_device()
    model = model.to(device)
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_macro_auroc": [], "val_macro_f1": []}

    print(f"Training on {device} for {epochs} epochs")
    try:
        for epoch in range(epochs):
            # ---- train ----
            model.train()
            running, n = 0.0, 0
            for ecg, tabular, labels in loaders["train"]:
                ecg, tabular = _to_model_input(ecg, tabular, device)
                labels = labels.to(device)

                optimizer.zero_grad()
                loss = criterion(model(ecg, tabular), labels)
                loss.backward()
                optimizer.step()

                running += loss.item() * labels.size(0)
                n += labels.size(0)
            train_loss = running / n

            # ---- validate ----
            y_true, y_prob = predict(model, loaders["valid"], device)
            val_loss = criterion(torch.as_tensor(y_prob), torch.as_tensor(y_true)).item()
            val_m = compute_metrics(y_true, y_prob)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_macro_auroc"].append(val_m["macro_auroc"])
            history["val_macro_f1"].append(val_m["macro_f1"])

            msg = (f"Epoch {epoch + 1:>2}/{epochs} | train loss {train_loss:.4f} | "
                   f"val loss {val_loss:.4f} | val macro AUROC {val_m['macro_auroc']:.4f} | "
                   f"val macro F1 {val_m['macro_f1']:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        **(extra_checkpoint_info or {}),
                    },
                    checkpoint_path,
                )
                msg += "  <- saved"
            print(msg)
    except KeyboardInterrupt:
        print("Training interrupted manually.")

    return history


def plot_history(history, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="valid")
    axes[0].set_title("BCE loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["val_macro_auroc"], label="macro AUROC")
    axes[1].plot(epochs, history["val_macro_f1"], label="macro F1 @0.5")
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    return fig


def load_model(checkpoint_path, device=None):
    """Rebuild the model from a checkpoint written by `train_model`."""
    device = device or get_device()
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = ECG1DCNN(ckpt["num_features"], ckpt["num_labels"])
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), ckpt
