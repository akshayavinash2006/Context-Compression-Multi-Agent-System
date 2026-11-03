"""Car plant scenarios (one agent per zone) and the live server for the 3D UI."""
from collections import Counter

import pytest

from agents.callbacks import CallLogger
from agents.graph import run_scenario
from agents.models import get_models
from data.automotive import ZONES, generate_plant
from data.simulator import validate
from graphmem.memory import GraphMemory


@pytest.fixture(scope="module")
def plant():
    s = generate_plant(1, seed=42)
    validate(s)
    return s


def test_plant_scenario_has_zone_agents(plant):
    assert set(plant["zones"]) == set(ZONES)
    sources = {e["source_agent"] for e in plant["events"] if e["type"] in ("sensor_reading", "operator_note")}
    assert sources and sources <= set(ZONES)          # shop-floor messages come from their zone's agent
    for e in plant["events"]:
        if e["type"] == "question":
            assert e["zone"] in ZONES and e["gold"]["machine"] in ZONES[e["zone"]]["machines"]


@pytest.mark.parametrize("mode", ["replay", "live"])
def test_plant_pipeline_runs_with_zone_agents(cfg, plant, mode):
    mem = GraphMemory(cfg)
    events = Counter()
    mem.listener = lambda kind, payload: events.update([kind])
    log = CallLogger()
    log.listener = lambda kind, payload: events.update([kind])
    out = run_scenario(plant, mem, get_models(cfg, "mock"), log, cfg, mode=mode)
    writers = {w for m, d in mem.store.g.nodes(data=True) if d["type"] == "memory" for w in mem.store.writers(m)}
    assert writers & set(ZONES)
    assert len(out["answers"]) == sum(e["type"] == "question" for e in plant["events"])
    assert events["write"] > 0 and events["read"] > 0 and events["llm"] > 0
    if mode == "live":
        agents = {t["agent"] for t in out["turns"]}
        assert agents & set(ZONES)                     # zone agents called the (mock) LLM themselves


def test_server_streams_a_run(tmp_path, monkeypatch, plant):
    import json

    from fastapi.testclient import TestClient

    import server.app as srv
    d = tmp_path / "plant"
    d.mkdir()
    (d / "P001.json").write_text(json.dumps(plant), encoding="utf-8")
    monkeypatch.setattr(srv, "SCENARIO_DIRS", {"plant": d})
    monkeypatch.setattr(srv, "CFG", srv.CFG.copy())
    srv.CFG.runtime.use_heavy_models = False
    srv.CFG.runtime.llm_cache = None
    c = TestClient(srv.app)
    assert c.get("/api/scenarios").json()[0]["has_zones"] is True
    seen = Counter()
    with c.websocket_connect("/ws/run") as ws:
        assert ws.receive_json()["type"] == "hello"
        ws.send_json({"type": "start", "scenario": "P001", "preset": "mock", "delay": 0})
        while True:
            m = ws.receive_json()
            seen[m["type"]] += 1
            if m["type"] == "run_started":
                rid = m["run_id"]
                assert "gold" not in json.dumps(m["scenario"]["events"])      # answer key is not leaked
            if m["type"] in ("done", "error"):
                break
    assert m["type"] == "done" and seen["answer"] >= 1 and seen["write"] > 0
    assert c.get(f"/api/runs/{rid}/memory/M1").json()["id"] == "M1"
    assert c.get(f"/api/runs/{rid}/trace/M1").json()["id"] == "M1"
    assert c.get(f"/api/runs/{rid}/graph").json()["nodes"]
