"""Experiment harness: conditions x scenarios x seeds (plan §16).

python -m eval.run_experiment --conditions full,window,summary,vector,graph \
       --scenarios data/scenarios/dev --seeds 42 --preset mock --out results/dev_mock

Ablations are conditions too: graph:no_ppr, graph:no_merge, graph:budget=100, ... (plan §18).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from agents.callbacks import CallLogger
from agents.graph import run_scenario, sqlite_checkpointer
from agents.models import describe, enable_llm_cache, get_models
from baselines import FullHistory, RollingSummary, SlidingWindow, VectorRAG
from graphmem.config import load_config
from graphmem.memory import GraphMemory, SharedModels
from graphmem.trace import leaf_event_ids, trace_size

ABLATIONS = {
    "no_compression": {"compression.rate": 1.0},
    "no_merge": {"merge.enabled": False},
    "no_ppr": {"retrieval.ranker": "keyword_count"},
    "no_strength": {"retrieval.use_strength": False},
    "no_read_skip": {"retrieval.skip_already_read": False},
    "no_push": {"notify.enabled": False},
    "no_mmr": {"retrieval.mmr_lambda": 1.0},
}


def parse_condition(cond: str) -> tuple[str, dict, int | None]:
    """'graph:no_ppr+no_mmr' or 'graph:budget=200' -> (backend, overrides, budget)."""
    name, _, mods = cond.partition(":")
    overrides, budget = {}, None
    for m in filter(None, mods.split("+")):
        if m.startswith("budget="):
            budget = int(m.split("=", 1)[1])
        elif m in ABLATIONS:
            overrides.update(ABLATIONS[m])
        else:
            raise ValueError(f"unknown modifier {m!r}; choose from {list(ABLATIONS)} or budget=N")
    return name, overrides, budget


def make_backend(name: str, cfg, models, logger, scenario_id: str):
    shared = SharedModels.get(cfg)
    if name == "graph":
        return GraphMemory(cfg, scenario_id=scenario_id)
    if name == "full":
        return FullHistory(cfg, shared["counter"])
    if name == "window":
        return SlidingWindow(cfg, shared["counter"])
    if name == "summary":
        return RollingSummary(cfg, shared["counter"], summariser=models["summariser"], logger=logger)
    if name == "vector":
        return VectorRAG(cfg, shared["embedder"], shared["counter"])
    raise ValueError(f"unknown condition {name}")


def load_scenarios(path: str, limit: int | None = None) -> list[dict]:
    p = Path(path)
    files = sorted(p.glob("*.json")) if p.is_dir() else [p]
    return [json.loads(f.read_text(encoding="utf-8")) for f in files][:limit]


def run_one(cond: str, scenario: dict, seed: int, cfg_base, preset: str, mode: str, out: Path,
            checkpointer=None, emissions: bool = False) -> dict:
    name, overrides, budget = parse_condition(cond)
    cfg = cfg_base.copy()
    for k, v in overrides.items():
        cfg.set_path(k, v)
    sid = scenario["scenario_id"]
    logger = CallLogger(condition=cond, scenario=sid, seed=seed, preset=preset)
    models = get_models(cfg, preset, seed=seed)
    backend = make_backend(name, cfg, models, logger, sid)
    tracker = None
    if emissions:
        try:
            from codecarbon import EmissionsTracker
            tracker = EmissionsTracker(project_name=f"{cond}-{sid}-{seed}", output_dir=str(out),
                                       log_level="error", save_to_file=True)
            tracker.start()
        except Exception:
            tracker = None
    t0 = time.perf_counter()
    final = run_scenario(scenario, backend, models, logger, cfg, mode=mode, checkpointer=checkpointer,
                         thread_id=f"{cond}-{sid}-{seed}", token_budget=budget)
    wall = time.perf_counter() - t0
    kg = tracker.stop() if tracker else None

    key = {"condition": cond, "scenario": sid, "seed": seed, "preset": preset}
    logger.dump(out / "calls.jsonl")
    with open(out / "answers.jsonl", "a", encoding="utf-8") as f:
        for a in final["answers"]:
            tr = backend.trace(a["report_memory_id"])
            a = {**key, **a, "trace_leaves": sorted(leaf_event_ids(tr)), "trace_size": trace_size(tr),
                 "history_size": backend.stats().get("writes", 0)}
            f.write(json.dumps(a, default=str) + "\n")
            if name == "graph":
                with open(out / "traces.jsonl", "a", encoding="utf-8") as g:
                    g.write(json.dumps({**key, "event_id": a["event_id"], "trace": tr}, default=str) + "\n")
    with open(out / "memory_stats.jsonl", "a", encoding="utf-8") as f:
        for m in final.get("maintenance_log", []):
            f.write(json.dumps({**key, **m}, default=str) + "\n")
        f.write(json.dumps({**key, "final": True, **backend.stats()}, default=str) + "\n")
    with open(out / "decisions.jsonl", "a", encoding="utf-8") as f:
        for d in getattr(backend, "decisions", []):
            f.write(json.dumps({**key, **d}, default=str) + "\n")
    with open(out / "turns.jsonl", "a", encoding="utf-8") as f:
        for t in final["turns"]:
            if not t.get("replay"):
                f.write(json.dumps({**key, **t}, default=str) + "\n")
    with open(out / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({**key, "wall_s": round(wall, 3), "emissions_kg": kg,
                            "models": {r: c.primary for r, c in models.items()}}) + "\n")
    return final


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="full,window,summary,vector,graph")
    ap.add_argument("--scenarios", default="data/scenarios/dev")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--mode", default=None, choices=["replay", "live"])
    ap.add_argument("--preset", default=None, help="mock | single_provider | multi_provider")
    ap.add_argument("--out", default="results/run")
    ap.add_argument("--limit", type=int, default=None, help="max scenarios")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides a.b=v")
    ap.add_argument("--checkpoint", action="store_true", help="save LangGraph checkpoints (SqliteSaver)")
    ap.add_argument("--emissions", action="store_true", help="CodeCarbon energy/CO2 per run")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.ERROR)
    import warnings
    # Some Gemini models ignore temperature; the provider warns on every call. Noted once here instead.
    warnings.filterwarnings("ignore", message=r".*uses fixed sampling defaults.*")

    cfg = load_config(args.config, args.set)
    preset = args.preset or cfg.agents.preset
    mode = args.mode or cfg.experiment.mode
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("calls", "answers", "traces", "memory_stats", "decisions", "turns", "runs", "failures"):
        (out / f"{f}.jsonl").unlink(missing_ok=True)
    cached = enable_llm_cache(cfg, preset)
    print(f"preset={preset} mode={mode} llm_cache={cached}")
    print(describe(get_models(cfg, preset)))
    checkpointer = sqlite_checkpointer(str(out / "checkpoints.sqlite")) if args.checkpoint else None

    scenarios = load_scenarios(args.scenarios, args.limit)
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    total = len(conds) * len(scenarios) * len(args.seeds)
    n = failed = 0
    for seed in args.seeds:
        for sc in scenarios:
            for cond in conds:
                n += 1
                t0 = time.perf_counter()
                try:
                    run_one(cond, sc, seed, cfg, preset, mode, out, checkpointer, args.emissions)
                    status = "ok"
                except Exception as e:   # no model left (quota, outage, missing key): skip, never fake
                    failed += 1
                    status = f"FAILED: {str(e)[:160]}"
                    with open(out / "failures.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps({"condition": cond, "scenario": sc["scenario_id"], "seed": seed,
                                            "error": f"{type(e).__name__}: {e}"}) + "\n")
                print(f"[{n}/{total}] {cond:<22} {sc['scenario_id']} seed={seed} "
                      f"({time.perf_counter() - t0:.1f}s) {status}")
    (out / "scenarios.json").write_text(json.dumps(
        {s["scenario_id"]: s for s in scenarios}), encoding="utf-8")
    if failed:
        print(f"\nWARNING: {failed}/{total} runs failed and are excluded (see {out / 'failures.jsonl'}).")
    if not args.no_report and failed < total:
        from eval.metrics import compute_all
        from eval.plots import make_plots
        table = compute_all(out)
        make_plots(out)
        print(table.to_string())
        import pandas as pd
        per_run = pd.read_csv(out / "per_run.csv")
        fb = per_run[per_run.fallback_calls > 0]
        if len(fb):
            print(f"\nWARNING: {len(fb)} runs used a fallback model for some calls "
                  f"(column fallback_calls in per_run.csv); their results mix models.")
        print(f"\nresults in {out}")


if __name__ == "__main__":
    main()
