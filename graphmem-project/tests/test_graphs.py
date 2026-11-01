"""LangGraph pipelines: write branches, read fallback branch, agent tool loop."""
from langchain_core.messages import ToolMessage

from agents.callbacks import CallLogger
from agents.graph import build_agent_graph
from agents.models import get_models
from tests.conftest import WORKED


def test_write_graph_branches(mem):
    out = mem._write_graph.invoke({"agent": "sensor", "text": WORKED["m1"], "derived_from": []})
    assert out["decision"] == "create"
    out = mem._write_graph.invoke({"agent": "sensor", "text": WORKED["dup"], "derived_from": []})
    assert out["decision"] == "merge" and out["memory_id"] == "M1"
    mem._write_graph.invoke({"agent": "history", "text": "CNC-3 belt delivery delayed by 10 days.", "derived_from": []})
    out = mem._write_graph.invoke({"agent": "history", "text": "CNC-3 belt delivery now delayed by 14 days.",
                                   "derived_from": []})
    assert out["decision"] == "update" and out["superseded"]


def test_read_graph_vector_fallback(mem):
    mem.write("sensor", WORKED["m1"])
    res = mem.read("diagnosis", "zzqx wpvk", record=False)
    assert res.fallback in ("vector", "empty") and not res.seeds_used


def test_read_graph_ppr_branch(mem):
    mem.write("sensor", WORKED["m1"])
    res = mem.read("diagnosis", "CNC-3 vibration", record=False)
    assert res.fallback is None and res.memory_ids == ["M1"]


def test_agent_tool_loop_calls_expand_once(cfg, mem):
    c = cfg.copy()
    c.agents.mock.expand_policy = "always"
    models = get_models(c, "mock")
    mem.write("sensor", WORKED["m1"], event_id="E1")
    mem.write("history", WORKED["m2"], event_id="E2")
    mem.tick()
    logger = CallLogger()
    agent = build_agent_graph()
    out = agent.invoke({"agent": "diagnosis", "event": {"id": "Q1", "text": "Why is CNC-3 vibrating?"}},
                       {"configurable": {"memory": mem, "models": models, "logger": logger, "cfg": c}})
    tool_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1 and out["n_tool_calls"] == 1      # capped at max_expand_per_turn
    assert "original" in tool_msgs[0].content
    assert mem.counters["expand"] == 1
    assert "bearing_wear" in out["answer"]
    assert len(logger.records) == 2                               # tool-call round + final answer
