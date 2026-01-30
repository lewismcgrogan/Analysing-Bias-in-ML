from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else float("nan")


def group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: pd.Series,
    reference_group: Optional[str] = None,
) -> pd.DataFrame:
    """
    Computes group fairness metrics:
    - selection_rate (P(\hat{Y}=1))
    - base_rate      (P(Y=1))
    - TPR/FPR/FNR
    - PPV/NPV
    - disparate_impact vs reference group (selection_rate_group / selection_rate_ref)
    """
    df = pd.DataFrame(
        {
            "y_true": y_true.astype(int),
            "y_pred": y_pred.astype(int),
            "group": sensitive.astype(str).fillna("NA"),
        }
    )

    if reference_group is None:
        # Default: majority group by count
        reference_group = df["group"].value_counts().idxmax()

    rows = []
    for g, sub in df.groupby("group"):
        yt = sub["y_true"].to_numpy()
        yp = sub["y_pred"].to_numpy()

        n = len(sub)
        tp = int(((yp == 1) & (yt == 1)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())

        selection_rate = float((yp == 1).mean()) if n else float("nan")
        base_rate = float((yt == 1).mean()) if n else float("nan")

        tpr = _safe_div(tp, tp + fn)  # recall for positives
        fpr = _safe_div(fp, fp + tn)
        fnr = _safe_div(fn, fn + tp)
        ppv = _safe_div(tp, tp + fp)  # precision
        npv = _safe_div(tn, tn + fn)

        rows.append(
            {
                "group": g,
                "n": n,
                "base_rate": base_rate,
                "selection_rate": selection_rate,
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
                "TPR": tpr,
                "FPR": fpr,
                "FNR": fnr,
                "PPV": ppv,
                "NPV": npv,
                "reference_group": reference_group,
            }
        )

    out = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)

    ref_sr = float(out.loc[out["group"] == reference_group, "selection_rate"].iloc[0]) if not out.empty else float("nan")
    out["disparate_impact"] = out["selection_rate"].apply(lambda sr: _safe_div(sr, ref_sr))
    out["four_fifths_rule_pass"] = out["disparate_impact"].apply(
        lambda di: (di >= 0.8) if np.isfinite(di) else False
    )
    return out


def intersectional_sensitive(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """
    Creates an intersectional group label like "sex=Male|race=White"
    """
    parts = []
    for c in cols:
        parts.append(df[c].astype(str).fillna("NA").map(lambda v: f"{c}={v}"))
    out = parts[0]
    for p in parts[1:]:
        out = out + "|" + p
    return out


def _best_threshold_to_match_rate(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_rate: float,
    grid: np.ndarray,
) -> float:
    # Choose threshold that makes selection_rate closest to target_rate
    best_t, best_gap = 0.5, float("inf")
    for t in grid:
        yp = (scores >= t).astype(int)
        sr = float(yp.mean())
        gap = abs(sr - target_rate)
        if gap < best_gap:
            best_gap = gap
            best_t = float(t)
    return best_t


def _best_threshold_to_match_tpr(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_tpr: float,
    grid: np.ndarray,
) -> float:
    best_t, best_gap = 0.5, float("inf")
    for t in grid:
        yp = (scores >= t).astype(int)
        # TPR
        positives = (y_true == 1)
        denom = positives.sum()
        tpr = float(((yp == 1) & positives).sum() / denom) if denom else float("nan")
        gap = abs(tpr - target_tpr) if np.isfinite(tpr) else float("inf")
        if gap < best_gap:
            best_gap = gap
            best_t = float(t)
    return best_t


@dataclass(frozen=True)
class GroupThresholds:
    method: str
    thresholds: Dict[str, float]
    reference_group: str


def fit_group_thresholds(
    y_true_val: np.ndarray,
    scores_val: np.ndarray,
    sensitive_val: pd.Series,
    method: str = "equal_opportunity",
    reference_group: Optional[str] = None,
    grid_size: int = 501,
) -> GroupThresholds:
    """
    Post-processing mitigation via group-specific thresholds.
    Methods:
      - demographic_parity: match selection rate of reference group
      - equal_opportunity: match TPR of reference group
    Fit on validation data, apply to test for honest reporting.
    """
    df = pd.DataFrame(
        {"y_true": y_true_val.astype(int), "score": scores_val, "group": sensitive_val.astype(str).fillna("NA")}
    )
    if reference_group is None:
        reference_group = df["group"].value_counts().idxmax()

    grid = np.linspace(0.0, 1.0, grid_size)

    ref = df[df["group"] == reference_group]
    if method == "demographic_parity":
        target_rate = float((ref["score"].to_numpy() >= 0.5).mean())
        # Better target: reference selection rate at threshold 0.5 on val
        thresholds = {}
        for g, sub in df.groupby("group"):
            thresholds[g] = _best_threshold_to_match_rate(
                sub["y_true"].to_numpy(), sub["score"].to_numpy(), target_rate, grid
            )
    elif method == "equal_opportunity":
        ref_scores = ref["score"].to_numpy()
        ref_true = ref["y_true"].to_numpy()
        # Reference TPR at threshold 0.5
        ref_pred = (ref_scores >= 0.5).astype(int)
        denom = int((ref_true == 1).sum())
        target_tpr = float(((ref_pred == 1) & (ref_true == 1)).sum() / denom) if denom else 0.0

        thresholds = {}
        for g, sub in df.groupby("group"):
            thresholds[g] = _best_threshold_to_match_tpr(
                sub["y_true"].to_numpy(), sub["score"].to_numpy(), target_tpr, grid
            )
    else:
        raise ValueError("method must be 'demographic_parity' or 'equal_opportunity'")

    return GroupThresholds(method=method, thresholds=thresholds, reference_group=str(reference_group))


def apply_group_thresholds(scores: np.ndarray, sensitive: pd.Series, group_thresholds: GroupThresholds) -> np.ndarray:
    groups = sensitive.astype(str).fillna("NA").to_numpy()
    y_pred = np.zeros_like(scores, dtype=int)
    for i, g in enumerate(groups):
        t = group_thresholds.thresholds.get(g, 0.5)
        y_pred[i] = int(scores[i] >= t)
    return y_pred
