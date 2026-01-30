from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import pandas as pd


def to_markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    d = df.copy()
    if len(d) > max_rows:
        d = d.head(max_rows)
    return d.to_markdown(index=False)


def write_report_artifacts(
    out_dir: str | Path,
    baseline_metrics: Dict[str, Any],
    mitigations_metrics: Dict[str, Any],
    group_tables: Dict[str, pd.DataFrame],
) -> None:
    out_dir = Path(out_dir)
    md_path = out_dir / "report_artifacts.md"

    lines = []
    lines.append("# Report Artifacts (Auto-generated)\n")

    lines.append("## Baseline performance\n")
    for k in ["accuracy", "f1", "precision", "recall", "roc_auc"]:
        lines.append(f"- **{k}**: {baseline_metrics.get(k)}")
    lines.append("")

    lines.append("## Mitigation performance summary\n")
    for name, m in mitigations_metrics.items():
        lines.append(f"### {name}")
        for k in ["accuracy", "f1", "precision", "recall", "roc_auc"]:
            lines.append(f"- **{k}**: {m.get(k)}")
        lines.append("")

    lines.append("## Fairness tables (top groups by size)\n")
    for name, df in group_tables.items():
        lines.append(f"### {name}\n")
        lines.append(to_markdown_table(df, max_rows=20))
        lines.append("")

    lines.append("## Suggested report bullets (edit into your own words)\n")
    lines.append(
        "- Baseline model may be procedurally reasonable (same threshold for all), yet distributively uneven outcomes can appear across protected groups.\n"
        "- Disparate impact is assessed via the 4/5ths rule (DI < 0.8 indicates potential adverse impact).\n"
        "- Post-processing group thresholds can reduce specific fairness gaps (e.g., TPR gaps / selection-rate gaps) but may trade off with global accuracy/precision.\n"
        "- Intersectional evaluation (sex×race) is included to avoid hiding harms that do not appear in single-attribute analysis.\n"
    )

    md_path.write_text("\n".join(lines), encoding="utf-8")
