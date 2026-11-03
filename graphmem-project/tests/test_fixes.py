"""Regression tests for: machine-name scoring, reasoning stripping, no silent mock
fallback, seed-scoped cache, and scenario-level statistics."""
import json

import pandas as pd
import pytest
from langchain_core.caches import InMemoryCache

from agents.graph import strip_reasoning, _text
from agents.models import SeedScopedCache, get_models
from agents.schemas import parse_report_text
from eval.metrics import run_metrics
from eval.stats import compare
from graphmem.models import normalize_machine


# ---- 1. machine names ----
@pytest.mark.parametrize("raw", ["cnc-3", "CNC-3", "CNC 3", "cnc3", "CNC_3", "Machine CNC-3", " cnc-03 "])
def test_normalize_machine(raw):
    assert normalize_machine(raw) == "cnc-3"


def test_report_parse_and_scoring_normalise_machine():
    rep = parse_report_text('{"machine": "LATHE4", "root_cause": "Bearing Wear", "action": "x", "evidence": ["[M1]"]}')
    assert rep.machine == "lathe-4" and rep.root_cause == "bearing_wear" and rep.evidence == ["M1"]
    ans = [{"event_id": "Q", "report": {"machine": "CNC 3", "root_cause": "bearing_wear", "action": ""},
            "gold": {"machine": "cnc-3", "root_cause": "bearing_wear", "action": "", "supporting_events": []},
            "hops": 2}]
    r = run_metrics(("graph", "S", 42), [], ans, [], [], [], [], {"events": [], "injected_error": None})
    assert r["accuracy"] == 1.0


# ---- 2. reasoning text ----
def test_strip_reasoning():
    assert strip_reasoning("<think>cnc-5 maybe [M9]</think>Likely bearing_wear [M1].") == "Likely bearing_wear [M1]."
    assert strip_reasoning("<THINK>\nlong\n</THINK>\n answer") == "answer"
    assert strip_reasoning("answer <think>cut off mid-thought [M7]") == "answer"
    assert _text([{"type": "thinking", "thinking": "[M9]"}, {"type": "text", "text": "ok [M1]"}]) == "ok [M1]"


# ---- 3. no silent mock fallback ----
def _real_only_cfg(cfg, allow_mock=False):
    c = cfg.copy()
    c.agents.presets["test_real"] = {"default": {"provider": "groq", "model": "does-not-matter"}}
    c.agents.fallbacks = [{"provider": "mock", "model": "rule-based"}]
    c.agents.allow_mock_fallback = allow_mock
    return c


def test_real_preset_never_uses_mock(cfg, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")            # key missing -> groq skipped
    chains = get_models(_real_only_cfg(cfg), "test_real")
    ch = chains["diagnosis"]
    assert ch.specs == [] and ch.primary == "UNAVAILABLE"
    with pytest.raises(RuntimeError, match="No model available"):
        ch.invoke(["hi"])


def test_mock_fallback_only_when_allowed(cfg, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    ch = get_models(_real_only_cfg(cfg, allow_mock=True), "test_real")["diagnosis"]
    assert [s.provider for s in ch.specs] == ["mock"]


def test_harness_records_failed_runs(tmp_path, monkeypatch):
    from data.simulator import worked_example
    from eval.run_experiment import main
    monkeypatch.setenv("GROQ_API_KEY", "")
    sc = tmp_path / "w.json"
    sc.write_text(json.dumps(worked_example()), encoding="utf-8")
    main(["--conditions", "graph", "--scenarios", str(sc), "--preset", "single_provider", "--out", str(tmp_path / "o"),
          "--set", "runtime.use_heavy_models=false", "runtime.llm_cache=null", "agents.fallbacks=[]",
          "agents.presets.single_provider.default={provider: groq, model: x}"])
    fails = (tmp_path / "o" / "failures.jsonl").read_text().splitlines()
    assert len(fails) == 1 and "No model available" in fails[0]
    answers = tmp_path / "o" / "answers.jsonl"
    assert not answers.exists() or answers.read_text() == ""        # nothing faked


# ---- 5. seeds ----
def test_seed_scoped_cache():
    base = InMemoryCache()
    a, b = SeedScopedCache(base, 42), SeedScopedCache(base, 43)
    a.update("prompt", "llm", ["gen"])
    assert a.lookup("prompt", "llm") == ["gen"]
    assert b.lookup("prompt", "llm") is None


def test_stats_pair_by_scenario_not_seed():
    rows = []
    for cond, tok in (("graph", 100), ("full", 300)):
        for s in range(6):
            for seed in (42, 43, 44):
                rows.append({"condition": cond, "scenario": f"S{s}", "seed": seed,
                             "tokens_per_task": tok + s, "accuracy": 1.0})
    out = compare(pd.DataFrame(rows))
    row = out[(out.baseline == "full") & (out.metric == "tokens_per_task")].iloc[0]
    assert row["n"] == 6
