from agents.callbacks import CallLogger
from agents.graph import run_scenario
from agents.models import get_models
from data.simulator import worked_example
from graphmem.memory import GraphMemory
from graphmem.trace import iter_nodes, leaf_event_ids
from tests.conftest import WORKED


def test_trace_of_diagnosis(mem):
    m1 = mem.write("sensor", WORKED["m1"], event_id="E004")
    m2 = mem.write("history", WORKED["m2"], event_id="E006")
    m4 = mem.write("diagnosis", f"Likely worn spindle bearing, 2 months overdue [{m1}] [{m2}]",
                   derived_from=[m1, m2], event_id="E009")
    tr = mem.trace(m4)
    assert tr["kind"] == "derived" and tr["agents"] == ["diagnosis"]
    src = {s["id"]: s for s in tr["sources"]}
    assert set(src) == {m1, m2}
    assert src[m1]["agents"] == ["sensor"] and src[m2]["agents"] == ["history"]
    assert src[m1]["originals"][0]["text"] == WORKED["m1"]
    assert leaf_event_ids(tr) == {"E004", "E006"}


def test_merged_memory_returns_all_originals(mem):
    m1 = mem.write("sensor", WORKED["m1"], event_id="E1")
    mem.write("sensor", WORKED["dup"], event_id="E2")
    assert {o["event_id"] for o in mem.trace(m1)["originals"]} == {"E1", "E2"}


def test_trace_through_pipeline(cfg):
    mem = GraphMemory(cfg)
    out = run_scenario(worked_example(), mem, get_models(cfg, "mock"), CallLogger(), cfg)
    a = out["answers"][0]
    assert a["report"]["root_cause"] == "bearing_wear"
    tr = mem.trace(a["report_memory_id"])
    agents = {ag for n in iter_nodes(tr) for ag in n["agents"]}
    assert {"report", "diagnosis", "sensor", "history"} <= agents
    assert {"E003", "E005"} <= leaf_event_ids(tr)
