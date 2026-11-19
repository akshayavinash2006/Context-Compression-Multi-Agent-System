"""Paired Wilcoxon signed-rank tests: ours vs each baseline, per metric (plan §17)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRICS = ["tokens_per_task", "accuracy", "multihop_acc", "unnecessary_context", "latency_per_question_s",
           "trace_precision", "trace_recall", "stored_tokens"]


def compare(df: pd.DataFrame, ours: str = "graph") -> pd.DataFrame:
    """Pairs are scenarios: each metric is first averaged over seeds, so repeated seeds of the
    same scenario never count as independent samples."""
    rows = []
    cols = [m for m in METRICS if m in df]
    per_sc = df.groupby(["condition", "scenario"])[cols].mean().reset_index()
    base = per_sc[per_sc.condition == ours].set_index("scenario")
    for cond in sorted(set(df.condition) - {ours}):
        other = per_sc[per_sc.condition == cond].set_index("scenario")
        idx = base.index.intersection(other.index)
        for m in cols:
            a, b = base.loc[idx, m].astype(float), other.loc[idx, m].astype(float)
            ok = a.notna() & b.notna()
            a, b = a[ok].values, b[ok].values
            row = {"baseline": cond, "metric": m, "n": len(a), "ours_mean": np.mean(a) if len(a) else np.nan,
                   "baseline_mean": np.mean(b) if len(b) else np.nan, "p_value": np.nan, "effect_r": np.nan}
            d = a - b
            if len(d) >= 5 and np.any(d != 0):
                res = wilcoxon(a, b, zero_method="wilcox", method="auto")
                row["p_value"] = float(res.pvalue)
                # rank-biserial correlation as effect size
                nz = d[d != 0]
                ranks = pd.Series(np.abs(nz)).rank().values
                row["effect_r"] = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())
            rows.append(row)
    return pd.DataFrame(rows)
