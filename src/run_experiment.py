from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from src.data import load_adult, split_features_target
from src.model import train_model, predict_scores, predict_labels, eval_binary, save_model
from src.fairness import (
    group_metrics,
    intersectional_sensitive,
    fit_group_thresholds,
    apply_group_thresholds,
)
from src.plots import ensure_dir, plot_roc, plot_confusion, plot_group_bars, plot_tradeoff
from src.report_artifacts import write_report_artifacts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True, help="Path to data/ containing adult.data and adult.test")
    p.add_argument("--out_dir", type=str, required=True, help="Output base directory, e.g. results/runs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_size", type=float, default=0.2, help="Validation size from training split for mitigation fit")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dataset = load_adult(args.data_dir)
    X_train_full, y_train_full = split_features_target(dataset.train)
    X_test, y_test = split_features_target(dataset.test)

    # Split train into train/val so mitigation thresholds are fit honestly
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=y_train_full,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / run_id
    fig_dir = ensure_dir(run_dir / "figures")

    # Train baseline model
    model = train_model(X_tr, y_tr, seed=args.seed)

    # Baseline on test
    test_scores = predict_scores(model, X_test)
    test_pred = predict_labels(test_scores, threshold=0.5)
    baseline_metrics = eval_binary(y_test.to_numpy(), test_pred, test_scores)

    # Group fairness tables (baseline)
    sex_table = group_metrics(y_test.to_numpy(), test_pred, X_test["sex"], reference_group="Male" if "Male" in set(X_test["sex"].astype(str)) else None)
    race_table = group_metrics(y_test.to_numpy(), test_pred, X_test["race"], reference_group="White" if "White" in set(X_test["race"].astype(str)) else None)
    inter = intersectional_sensitive(X_test, ["sex", "race"])
    inter_table = group_metrics(y_test.to_numpy(), test_pred, inter, reference_group=None)

    # Plots (baseline)
    plot_roc(y_test.to_numpy(), test_scores, fig_dir / "roc_baseline.png")
    plot_confusion(y_test.to_numpy(), test_pred, fig_dir / "cm_baseline.png")
    plot_group_bars(sex_table, "selection_rate", "Baseline selection rate by sex", fig_dir / "baseline_selection_sex.png")
    plot_group_bars(race_table, "selection_rate", "Baseline selection rate by race", fig_dir / "baseline_selection_race.png")
    plot_group_bars(sex_table, "TPR", "Baseline TPR by sex", fig_dir / "baseline_tpr_sex.png")
    plot_group_bars(race_table, "TPR", "Baseline TPR by race", fig_dir / "baseline_tpr_race.png")

    # Mitigations: group thresholding (fit on validation, evaluate on test)
    val_scores = predict_scores(model, X_val)

    mitigations = {}

    # Equal opportunity on sex
    gt_eo_sex = fit_group_thresholds(
        y_true_val=y_val.to_numpy(),
        scores_val=val_scores,
        sensitive_val=X_val["sex"],
        method="equal_opportunity",
        reference_group="Male" if "Male" in set(X_val["sex"].astype(str)) else None,
    )
    eo_sex_pred = apply_group_thresholds(test_scores, X_test["sex"], gt_eo_sex)
    mitigations["equal_opportunity_sex"] = eval_binary(y_test.to_numpy(), eo_sex_pred, test_scores)

    # Demographic parity on sex
    gt_dp_sex = fit_group_thresholds(
        y_true_val=y_val.to_numpy(),
        scores_val=val_scores,
        sensitive_val=X_val["sex"],
        method="demographic_parity",
        reference_group="Male" if "Male" in set(X_val["sex"].astype(str)) else None,
    )
    dp_sex_pred = apply_group_thresholds(test_scores, X_test["sex"], gt_dp_sex)
    mitigations["demographic_parity_sex"] = eval_binary(y_test.to_numpy(), dp_sex_pred, test_scores)

    # Equal opportunity on race
    gt_eo_race = fit_group_thresholds(
        y_true_val=y_val.to_numpy(),
        scores_val=val_scores,
        sensitive_val=X_val["race"],
        method="equal_opportunity",
        reference_group="White" if "White" in set(X_val["race"].astype(str)) else None,
    )
    eo_race_pred = apply_group_thresholds(test_scores, X_test["race"], gt_eo_race)
    mitigations["equal_opportunity_race"] = eval_binary(y_test.to_numpy(), eo_race_pred, test_scores)

    # Demographic parity on race
    gt_dp_race = fit_group_thresholds(
        y_true_val=y_val.to_numpy(),
        scores_val=val_scores,
        sensitive_val=X_val["race"],
        method="demographic_parity",
        reference_group="White" if "White" in set(X_val["race"].astype(str)) else None,
    )
    dp_race_pred = apply_group_thresholds(test_scores, X_test["race"], gt_dp_race)
    mitigations["demographic_parity_race"] = eval_binary(y_test.to_numpy(), dp_race_pred, test_scores)

    # Save group tables for each variant (baseline + best mitigations)
    tables = {
        "baseline_by_sex.csv": sex_table,
        "baseline_by_race.csv": race_table,
        "baseline_by_sex_race.csv": inter_table,
        "eo_sex_by_sex.csv": group_metrics(y_test.to_numpy(), eo_sex_pred, X_test["sex"], reference_group=gt_eo_sex.reference_group),
        "dp_sex_by_sex.csv": group_metrics(y_test.to_numpy(), dp_sex_pred, X_test["sex"], reference_group=gt_dp_sex.reference_group),
        "eo_race_by_race.csv": group_metrics(y_test.to_numpy(), eo_race_pred, X_test["race"], reference_group=gt_eo_race.reference_group),
        "dp_race_by_race.csv": group_metrics(y_test.to_numpy(), dp_race_pred, X_test["race"], reference_group=gt_dp_race.reference_group),
    }

    for filename, df in tables.items():
        df.to_csv(run_dir / filename, index=False)

    # Trade-off plot summary (simple: accuracy vs worst DI)
    def worst_di(df: pd.DataFrame) -> float:
        return float(df["disparate_impact"].min())

    summary_rows = []
    summary_rows.append(
        {
            "variant": "baseline",
            "accuracy": baseline_metrics["accuracy"],
            "roc_auc": baseline_metrics["roc_auc"],
            "worst_group_di": worst_di(sex_table),
        }
    )
    for name, m in mitigations.items():
        # choose sex DI if sex mitigation, race DI if race mitigation (report-friendly)
        if "sex" in name:
            di = worst_di(tables["eo_sex_by_sex.csv"] if "equal_opportunity_sex" == name else tables["dp_sex_by_sex.csv"])
        else:
            di = worst_di(tables["eo_race_by_race.csv"] if "equal_opportunity_race" == name else tables["dp_race_by_race.csv"])
        summary_rows.append(
            {
                "variant": name,
                "accuracy": m["accuracy"],
                "roc_auc": m["roc_auc"],
                "worst_group_di": di,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(run_dir / "tradeoff_summary.csv", index=False)
    plot_tradeoff(summary, fig_dir / "tradeoff_accuracy_vs_di.png")

    # Save metrics
    (run_dir / "metrics_baseline.json").write_text(json.dumps(baseline_metrics, indent=2), encoding="utf-8")
    (run_dir / "metrics_mitigations.json").write_text(json.dumps(mitigations, indent=2), encoding="utf-8")

    # Save model
    save_model(model, str(run_dir / "model.joblib"))

    # Auto-generate report artifacts (markdown tables + bullets)
    group_tables_for_md = {
        "Baseline: Sex": sex_table,
        "Baseline: Race": race_table,
        "Baseline: Sex×Race (intersection)": inter_table,
    }
    write_report_artifacts(
        out_dir=run_dir,
        baseline_metrics=baseline_metrics,
        mitigations_metrics=mitigations,
        group_tables=group_tables_for_md,
    )

    print(f"\n✅ Run complete. Outputs saved to: {run_dir}\n")
    print(f"Key figures in: {run_dir / 'figures'}")
    print(f"Report paste-ready file: {run_dir / 'report_artifacts.md'}")


if __name__ == "__main__":
    main()
