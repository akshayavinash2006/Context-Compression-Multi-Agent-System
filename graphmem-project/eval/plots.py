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

    # Keep end-to-end wall time separate from summed LLM call time: the latter
    # excludes orchestration, memory operations, and time between model calls.
    if not calls.empty and "latency_s" in calls:
        agent_latency = calls.groupby(["condition", "agent"], dropna=False)["latency_s"].sum().unstack(fill_value=0)
        run_latency = pd.DataFrame(read_jsonl(out / "runs.jsonl"))
        if not run_latency.empty and "wall_s" in run_latency:
            run_latency = run_latency.groupby("condition")["wall_s"].mean()
            latency = agent_latency.copy()
            latency["Pipeline wall time"] = run_latency.reindex(latency.index).fillna(0)
            ax = latency.plot.bar(figsize=(9, 4), width=0.8)
            ax.set_ylabel("seconds (sum of runs)")
            ax.set_title("Pipeline and agent latency")
            ax.legend(title="Agent / measurement", bbox_to_anchor=(1.02, 1), loc="upper left")
            ax.figure.tight_layout()
            ax.figure.savefig(plots / "latency.png", dpi=130)
            plt.close(ax.figure)

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


def write_latency_report(out: str | Path):
    """Write latency results with end-to-end and per-agent measurements."""
    out = Path(out)
    calls = pd.DataFrame(read_jsonl(out / "calls.jsonl"))
    runs = pd.DataFrame(read_jsonl(out / "runs.jsonl"))
    lines = ["# Pipeline latency results", "",
             "Pipeline wall time measures the complete scenario run. Agent latency sums recorded LLM calls; it excludes orchestration, memory work, and gaps between calls.", ""]
    if calls.empty:
        lines.append("No LLM call latency records were available.")
    else:
        agent = calls.groupby(["condition", "agent"], dropna=False)["latency_s"].agg(["sum", "mean", "count"])
        lines += ["## Per-agent latency", "", "| Condition | Agent | Total call time (s) | Mean call time (s) | Calls |", "|---|---|---:|---:|---:|"]
        for (condition, name), row in agent.iterrows():
            lines.append(f"| {condition} | {name} | {row['sum']:.3f} | {row['mean']:.3f} | {int(row['count'])} |")
        pipe_calls = calls.groupby("condition")["latency_s"].sum()
        wall = runs.groupby("condition")["wall_s"].mean() if not runs.empty and "wall_s" in runs else pd.Series(dtype=float)
        lines += ["", "## Whole pipeline", "", "| Condition | Mean pipeline wall time (s) | Summed LLM call time (s) |", "|---|---:|---:|"]
        for condition in sorted(set(pipe_calls.index) | set(wall.index)):
            w = wall.get(condition, float("nan"))
            c = pipe_calls.get(condition, float("nan"))
            lines.append(f"| {condition} | {w:.3f} | {c:.3f} |")
        lines += ["", "![Pipeline and per-agent latency](plots/latency.png)"]
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
