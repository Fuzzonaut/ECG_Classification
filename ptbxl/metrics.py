"""Evaluation metrics for multi-label ECG classification.

Why not "accuracy"?
-------------------
The original notebook reported ``(preds == labels).sum() / labels.numel()``,
i.e. the fraction of *individual label cells* that are correct. With 44
labels and on average only ~1.2 positives per record, ~97% of cells are 0,
so a model that predicts "no disease" for everything already scores ~97%.
That number says almost nothing about diagnostic ability.

Instead we report the metrics used in the PTB-XL benchmark literature
(Strodthoff et al., 2020) plus standard multi-label metrics:

* macro AUROC          - threshold-free, every label weighted equally (main metric)
* micro AUROC
* macro / micro average precision (area under PR curve; sensitive to rare labels)
* macro / micro / sample F1 at a fixed threshold
* subset accuracy      - the whole label vector must match exactly
* Hamming loss
* label-wise accuracy  - the old metric, shown next to the all-negative baseline
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _evaluable(y_true):
    """Labels that have both positive and negative examples (AUROC is defined)."""
    pos = y_true.sum(axis=0)
    return (pos > 0) & (pos < y_true.shape[0])


def compute_metrics(y_true, y_prob, threshold=0.5):
    """Summary metrics as a flat dict."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    ok = _evaluable(y_true)

    m = {
        "macro_auroc": roc_auc_score(y_true[:, ok], y_prob[:, ok], average="macro"),
        "micro_auroc": roc_auc_score(y_true, y_prob, average="micro"),
        "macro_ap": average_precision_score(y_true[:, ok], y_prob[:, ok], average="macro"),
        "micro_ap": average_precision_score(y_true, y_prob, average="micro"),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "samples_f1": f1_score(y_true, y_pred, average="samples", zero_division=0),
        "subset_accuracy": float((y_pred == y_true).all(axis=1).mean()),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "labelwise_accuracy": float((y_pred == y_true).mean()),
        "labelwise_accuracy_all_negative_baseline": float((y_true == 0).mean()),
        "threshold": threshold,
        "n_samples": int(y_true.shape[0]),
        "n_labels": int(y_true.shape[1]),
        "n_labels_evaluated_for_auc": int(ok.sum()),
    }
    return {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in m.items()}


def per_label_report(y_true, y_prob, label_names, threshold=0.5):
    """One row per label: support, AUROC, AP, precision, recall, F1."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    rows = []
    for j, name in enumerate(label_names):
        t, p, yhat = y_true[:, j], y_prob[:, j], y_pred[:, j]
        both = 0 < t.sum() < len(t)
        rows.append(
            {
                "label": name,
                "support": int(t.sum()),
                "auroc": roc_auc_score(t, p) if both else np.nan,
                "average_precision": average_precision_score(t, p) if t.sum() > 0 else np.nan,
                "precision": precision_score(t, yhat, zero_division=0),
                "recall": recall_score(t, yhat, zero_division=0),
                "f1": f1_score(t, yhat, zero_division=0),
            }
        )
    return pd.DataFrame(rows).set_index("label").sort_values("support", ascending=False)


def format_metrics(m):
    """Pretty, aligned text block for printing."""
    order = [
        ("macro_auroc", "Macro AUROC"),
        ("micro_auroc", "Micro AUROC"),
        ("macro_ap", "Macro avg. precision"),
        ("micro_ap", "Micro avg. precision"),
        ("macro_f1", "Macro F1"),
        ("micro_f1", "Micro F1"),
        ("samples_f1", "Sample-wise F1"),
        ("subset_accuracy", "Subset (exact-match) acc."),
        ("hamming_loss", "Hamming loss"),
        ("labelwise_accuracy", "Label-wise accuracy (old)"),
        ("labelwise_accuracy_all_negative_baseline", "  ...all-negative baseline"),
    ]
    lines = [f"{label:<28} {m[key]:.4f}" for key, label in order]
    lines.append(
        f"(threshold={m['threshold']}, n={m['n_samples']}, "
        f"AUROC over {m['n_labels_evaluated_for_auc']}/{m['n_labels']} labels)"
    )
    return "\n".join(lines)
