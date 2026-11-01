from tests.conftest import WORKED


def last_decision(mem):
    return mem.decisions[-1]["decision"]


def test_paraphrased_duplicate_merges(mem):
    m1 = mem.write("sensor", WORKED["m1"], event_id="E1")
    mem.tick()
    m = mem.write("sensor", WORKED["dup"], event_id="E2")
    assert m == m1 and last_decision(mem) == "merge"
    assert len(mem.expand(m1)) == 2                      # both originals kept
    assert mem.store.g.nodes[m1]["base_strength"] > 1.0


def test_contradiction_updates_and_supersedes(mem):
    old = mem.write("history", "Supplier delivery of spindle bearing for CNC-3 delayed by 10 days.")
    mem.tick()
    new = mem.write("history", "Supplier delivery of spindle bearing for CNC-3 now delayed by 14 days.")
    assert new != old and last_decision(mem) == "update"
    assert mem.store.has_edge(new, old, "supersedes")
    assert mem.store.g.nodes[old]["valid_until"] is not None
    assert old not in mem.store.view


def test_unrelated_creates(mem):
    a = mem.write("sensor", WORKED["m1"])
    b = mem.write("history", WORKED["unrelated"])
    assert a != b and last_decision(mem) == "create"


def test_different_machines_never_merge(mem):
    a = mem.write("sensor", "CNC-3 spindle vibration 7.2 mm/s, above the 6 mm/s limit.")
    b = mem.write("sensor", "CNC-5 spindle vibration 7.2 mm/s, above the 6 mm/s limit.")
    assert a != b and last_decision(mem) == "create"


def test_answer_never_supersedes_its_sources(mem):
    m1 = mem.write("sensor", WORKED["m1"])
    m2 = mem.write("history", WORKED["m2"])
    d = mem.write("diagnosis", f"CNC-3 spindle bearing overdue, vibration 7.2 mm/s [{m1}] [{m2}]",
                  derived_from=[m1, m2])
    assert d not in (m1, m2)
    assert mem.store.is_active(m1) and mem.store.is_active(m2)
    assert set(mem.store.out(d, "derived_from")) == {m1, m2}


def test_merge_disabled_always_creates(cfg):
    from graphmem.memory import GraphMemory
    c = cfg.copy()
    c.merge.enabled = False
    mem = GraphMemory(c)
    a = mem.write("sensor", WORKED["m1"])
    b = mem.write("sensor", WORKED["dup"])
    assert a != b
