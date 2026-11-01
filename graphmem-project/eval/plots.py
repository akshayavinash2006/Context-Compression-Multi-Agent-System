"""Plots (plan §17): token growth, accuracy vs tokens (Pareto), ablations, storage over time."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from eval.metrics import read_jsonl  # noqa: E402


def make_plots(out: str | Path):
    out = Path(out)
    plots = out / "plots"
    plots.mkdir(exist_ok=True)
    calls = pd.DataFrame(read_jsonl(out / "calls.jsonl"))
    per_run = pd.read_csv(out / "per_run.csv") if (out / "per_run.csv").exists() else pd.DataFrame()

    if not calls.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for cond, g in calls[calls.agent != "summariser"].groupby("condition"):
            m = g.groupby("turn")["input_tokens"].mean()
            ax.plot(m.index, m.values, marker=".", label=cond)
        ax.set_xlabel("turn")
        ax.set_ylabel("input tokens per call")
        ax.set_title("Token growth")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots / "token_growth.png", dpi=130)
        plt.close(fig)

    if not per_run.empty:
        s = per_run.groupby("condition")[["tokens_per_task", "accuracy"]].mean()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(s.tokens_per_task, s.accuracy)
        for c, row in s.iterrows():
            ax.annotate(c, (row.tokens_per_task, row.accuracy), textcoords="offset points", xytext=(4, 4))
        ax.set_xlabel("input tokens per task")
        ax.set_ylabel("accuracy")
        ax.set_title("Accuracy vs cost")
        fig.tight_layout()
        fig.savefig(plots / "pareto.png", dpi=130)
        plt.close(fig)

        abl = per_run[per_run.condition.str.startswith("graph")]
        if abl.condition.nunique() > 1:
            s = abl.groupby("condition")[["accuracy", "tokens_per_task"]].mean().sort_index()
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            s.accuracy.plot.bar(ax=axes[0], title="accuracy")
            s.tokens_per_task.plot.bar(ax=axes[1], title="tokens per task")
            fig.tight_layout()
            fig.savefig(plots / "ablations.png", dpi=130)
            plt.close(fig)

    stats = pd.DataFrame(read_jsonl(out / "memory_stats.jsonl"))
    if not stats.empty and "turn" in stats:
        st = stats[stats.get("final", pd.Series(False, index=stats.index)).ne(True)]
        if not st.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            for cond, g in st.groupby("condition"):
                m = g.groupby("turn")["stored_tokens_active"].mean()
                ax.plot(m.index, m.values, marker=".", label=cond)
            ax.set_xlabel("turn")
            ax.set_ylabel("stored tokens (active)")
            ax.set_title("Storage over time")
            ax.legend()
            fig.tight_layout()
            fig.savefig(plots / "storage.png", dpi=130)
            plt.close(fig)
