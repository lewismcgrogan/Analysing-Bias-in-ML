from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, ConfusionMatrixDisplay, confusion_matrix


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_roc(y_true: np.ndarray, y_score: np.ndarray, out_path: str | Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_path: str | Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_group_bars(group_df: pd.DataFrame, metric: str, title: str, out_path: str | Path) -> None:
    # group_df columns: group, metric
    d = group_df.sort_values("n", ascending=False).head(20)
    plt.figure(figsize=(10, 5))
    plt.bar(d["group"].astype(str), d[metric].astype(float))
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_tradeoff(summary: pd.DataFrame, out_path: str | Path) -> None:
    """
    Expects a df with columns:
      - variant
      - accuracy
      - roc_auc
      - worst_group_di (min disparate_impact)
      - worst_group_tpr_gap (max abs(TPR - TPR_ref))
    """
    plt.figure(figsize=(10, 5))
    x = np.arange(len(summary))
    plt.bar(x - 0.2, summary["accuracy"], width=0.4, label="accuracy")
    plt.bar(x + 0.2, summary["worst_group_di"], width=0.4, label="worst_group_DI")
    plt.xticks(x, summary["variant"], rotation=30, ha="right")
    plt.title("Fairness–Performance Trade-off (higher DI is better; closer to 1)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
