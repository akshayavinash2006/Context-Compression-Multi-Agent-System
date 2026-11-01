import json

import pytest

from agents.callbacks import CallLogger
from agents.graph import build_orchestrator, register_agents, sqlite_checkpointer
from agents.models import get_models
from agents.schemas import ReportSchema
from data.simulator import generate, validate, worked_example
from eval.metrics import compute_all
from eval.run_experiment import run_one
from graphmem.memory import GraphMemory


@pytest.fixture(scope="module")
def small_scenario():
    s = generate(1, seed=7, split="dev")
    validate(s)
    s["events"] = [e for e in s["events"] if e["type"] != "question"][:20] + \
        [e for e in s["events"] if e["type"] == "question"][:1]
    return s


@pytest.mark.parametrize("cond", ["full", "window", "summary", "vector", "graph", "graph:no_ppr+budget=150"])
def test_every_condition_runs(cfg, tmp_path, cond):
    sc = worked_example()
    run_one(cond, sc, 42, cfg, "mock", "replay", tmp_path)
    answers = [json.loads(line) for line in open(tmp_path / "answers.jsonl", encoding="utf-8")]
    assert len(answers) == 1
    ReportSchema(**answers[0]["report"])
    calls = [json.loads(line) for line in open(tmp_path / "calls.jsonl", encoding="utf-8")]
    assert {c["agent"] for c in calls} >= {"diagnosis", "report"}
    assert all(c["input_tokens"] > 0 for c in calls)


def test_metrics_pipeline(cfg, tmp_path, small_scenario):
    for cond in ("full", "graph"):
        for sc in (worked_example(), small_scenario):
            run_one(cond, sc, 42, cfg, "mock", "replay", tmp_path)
    (tmp_path / "scenarios.json").write_text(json.dumps({s["scenario_id"]: s for s in
                                                         (worked_example(), small_scenario)}))
    table = compute_all(tmp_path)
    assert {"full", "graph"} <= set(table.index)
    assert table.loc["graph", "tokens_per_task_mean"] < table.loc["full", "tokens_per_task_mean"]
    assert (tmp_path / "per_run.csv").exists() and (tmp_path / "summary.csv").exists()


def test_live_mode_runs_all_agents(cfg, tmp_path):
    run_one("graph", worked_example(), 42, cfg, "mock", "live", tmp_path)
    calls = [json.loads(line) for line in open(tmp_path / "calls.jsonl", encoding="utf-8")]
    assert {"sensor", "history", "diagnosis", "report"} <= {c["agent"] for c in calls}


def test_checkpointer_records_history(cfg):
    sc = worked_example()
    mem = GraphMemory(cfg)
    register_agents(mem, sc["machines"])
    app = build_orchestrator(sqlite_checkpointer(":memory:"))
    config = {"configurable": {"thread_id": "t1", "memory": mem, "models": get_models(cfg, "mock"),
                               "logger": CallLogger(), "cfg": cfg, "mode": "replay"}, "recursion_limit": 200}
    final = app.invoke({"events": sc["events"], "i": 0, "turn": 0, "answers": [], "turns": [],
                        "maintenance_log": []}, config)
    assert final["answers"][0]["report"]["root_cause"] == "bearing_wear"
    history = list(app.get_state_history({"configurable": {"thread_id": "t1"}}))
    assert len(history) > len(sc["events"])
    assert app.get_state({"configurable": {"thread_id": "t1"}}).values["i"] == len(sc["events"])
