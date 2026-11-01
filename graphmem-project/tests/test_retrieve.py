from tests.conftest import WORKED


def worked_graph(mem):
    ids = {}
    for key in ("m1", "m2", "m3", "unrelated"):
        ids[key] = mem.write("sensor" if key != "m2" else "history", WORKED[key])
    return ids


def test_worked_example_ranking(mem):
    """Plan §2.3: seeds {cnc-3, vibration} -> M1 > M2 > M3 > unrelated."""
    ids = worked_graph(mem)
    from graphmem import retrieve
    seeds = {"K:cnc-3": 0.5, "K:vibration": 0.5}
    scores = dict(retrieve.ppr_rank(mem.store, seeds, mem.cfg, mem.now))
    s = lambda k: scores.get(ids[k], 0.0)  # noqa: E731
    assert s("m1") > s("m2") > s("m3") > s("unrelated")
    # through the full read path the two relevant facts come first
    res = mem.read("diagnosis", "Why is CNC-3 vibrating?", record=False)
    assert res.memory_ids[:2] == [ids["m1"], ids["m2"]]
    assert ids["unrelated"] not in res.memory_ids


def test_budget_never_exceeded(mem):
    for i in range(30):
        mem.write("sensor", f"CNC-{i % 4 + 1} spindle vibration {i}.5 mm/s at 10:{i:02d}, check bearing {i}.")
    for budget in (20, 60, 150):
        res = mem.read("diagnosis", "CNC-1 vibration bearing", token_budget=budget, record=False)
        assert res.tokens_used <= budget
        assert mem.count_tokens(res.context_text) <= budget


def test_already_read_skipped(mem):
    worked_graph(mem)
    first = mem.read("diagnosis", "Why is CNC-3 vibrating?")
    second = mem.read("diagnosis", "Why is CNC-3 vibrating?")
    assert first.memory_ids and not set(first.memory_ids) & set(second.memory_ids)
    assert all(mem.store.has_edge(m, "A:diagnosis", "read_by") for m in first.memory_ids)


def test_read_skip_expires_after_window(mem):
    worked_graph(mem)
    first = mem.read("diagnosis", "Why is CNC-3 vibrating?")
    for _ in range(mem.cfg.retrieval.skip_read_window + 1):
        mem.tick()
    again = mem.read("diagnosis", "Why is CNC-3 vibrating?")
    assert set(first.memory_ids) & set(again.memory_ids)


def test_superseded_excluded(mem):
    old = mem.write("history", "Supplier delivery of spindle bearing for CNC-3 delayed by 10 days.")
    new = mem.write("history", "Supplier delivery of spindle bearing for CNC-3 now delayed by 14 days.")
    res = mem.read("history", "CNC-3 spindle bearing delivery delay", record=False)
    assert new in res.memory_ids and old not in res.memory_ids


def test_inbox_push(mem):
    mem.register_agent("diagnosis", "Diagnosis", ["vibration", "spindle"])
    m = mem.write("sensor", WORKED["m1"])
    assert m in mem.store.agent("diagnosis")["inbox"]
    assert mem.store.has_edge(m, "A:diagnosis", "sent_to")
    mem.read("diagnosis", "anything new on CNC-3?")
    assert m not in mem.store.agent("diagnosis")["inbox"]
