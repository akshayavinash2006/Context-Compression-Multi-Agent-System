"""Run a same-model baseline comparison using local Ollama qwen3:8b.

Example: python -m eval.benchmark_qwen3 --scenarios data/scenarios/dev --limit 2
"""
from __future__ import annotations

import argparse

from agents.models import get_models
from graphmem.config import load_config

PRESET = "qwen3_local"
MODEL = "ollama:qwen3:8b"


def verify_local_model(cfg) -> None:
    chains = get_models(cfg, PRESET, seed=42)
    roles = ("sensor", "history", "diagnosis", "report", "summariser")
    wrong = {role: chains[role].primary for role in roles if chains[role].primary != MODEL}
    if wrong:
        details = ", ".join(f"{role}={provider}" for role, provider in wrong.items())
        raise SystemExit(
            f"Benchmark cancelled: expected {MODEL} for every role, got {details}. "
            "Start Ollama and run `ollama pull qwen3:8b`; this benchmark will not mix models."
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", default="data/scenarios/dev")
    parser.add_argument("--out", default="results/qwen3_8b")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--conditions", default="full,graph",
                        help="same-model comparison; default compares full history with GraphMem")
    parser.add_argument("--mode", choices=["replay", "live"], default="replay")
    parser.add_argument("--emissions", action="store_true")
    parser.add_argument("--set", nargs="*", default=[], help="normal experiment config overrides")
    args = parser.parse_args(argv)

    # A benchmark run must not silently mix in the global qwen2.5 fallback.
    overrides = [*args.set, "agents.fallbacks=[]"]
    cfg = load_config(overrides=overrides)
    verify_local_model(cfg)

    # Reuse the established harness and metrics; the run logs include model names,
    # exact scenario/seed, call latency, token totals, and answer quality metrics.
    from eval.run_experiment import main as run
    run(["--conditions", args.conditions, "--scenarios", args.scenarios,
         "--seeds", *map(str, args.seeds), "--mode", args.mode,
         "--preset", PRESET, "--out", args.out, *(["--limit", str(args.limit)] if args.limit else []),
         *( ["--emissions"] if args.emissions else []), "--set", *overrides])


if __name__ == "__main__":
    main()
