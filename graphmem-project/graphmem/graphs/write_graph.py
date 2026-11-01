"""Write path W1-W4 as a LangGraph StateGraph (plan §8).

START -> store_original -> compress -> extract_keywords -> score_candidates
      -> (merge | update | create) -> link_notify -> END
"""
from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from graphmem import merge as merge_mod
from graphmem import notify


class WriteState(TypedDict, total=False):
    agent: str
    text: str
    recipients: list[str]
    derived_from: list[str]
    event_id: str
    kind: str
    cold_id: int
    compressed: str
    comp_meta: dict
    keyword_ids: list[str]
    candidates: list[dict]
    best: dict
    decision: Literal["merge", "update", "create"]
    memory_id: str
    superseded: str
    notified: list[str]


def build_write_graph(mem):
    def store_original(s: WriteState):
        return {"cold_id": mem.cold.add(s["agent"], mem.clock.now, s["text"], s.get("event_id"))}

    def compress(s: WriteState):                                   # W1
        comp, meta = mem.compressor(s["text"])
        if meta.get("fallback"):
            mem.counters["compress_fallbacks"] += 1
        return {"compressed": comp, "comp_meta": meta}

    def extract_keywords(s: WriteState):                           # W2
        terms = mem.extractor.extract(s["text"])
        kids = []
        for t in terms:
            kid, _ = mem.index.resolve(t.canonical, t.kind, create=True, surface=t.surface)
            if kid and kid not in kids:
                kids.append(kid)
        return {"keyword_ids": kids}

    def score_candidates(s: WriteState):                           # W3 scores
        decision, best, cands = merge_mod.score_candidates(mem, s["compressed"], s["keyword_ids"],
                                                           exclude=s.get("derived_from"))
        return {"decision": decision, "best": best or {}, "candidates": cands[:5]}

    def do_merge(s: WriteState):                                   # W3 actions
        target = s["best"]["id"]
        d = mem.store.g.nodes[target]
        d["base_strength"] = min(mem.cfg.strength.max, d["base_strength"] + mem.cfg.merge.merge_strength_boost)
        mem.counters["merge"] += 1
        return {"memory_id": target}

    def do_update(s: WriteState):
        old = s["best"]["id"]
        mid = mem._create_memory(s)
        mem.store.add_edge(mid, old, "supersedes", turn=mem.clock.now)
        mem.store.g.nodes[old]["valid_until"] = mem.clock.now
        mem.store.deactivate(old)
        mem.counters["update"] += 1
        return {"memory_id": mid, "superseded": old}

    def do_create(s: WriteState):
        mid = mem._create_memory(s)
        mem.counters["create"] += 1
        return {"memory_id": mid}

    def link_notify(s: WriteState):                                # W4
        mid, agent, store = s["memory_id"], s["agent"], mem.store
        if f"A:{agent}" not in store.g:
            store.add_agent(agent, agent, {})
        if not store.has_edge(mid, f"A:{agent}", "written_by"):
            store.add_edge(mid, f"A:{agent}", "written_by", turn=mem.clock.now)
        store.g.nodes[mid]["n_writers"] = len(set(store.writers(mid)))
        for p in s.get("derived_from") or []:
            if p != mid and p in store.g and not store.has_edge(mid, p, "derived_from"):
                store.add_edge(mid, p, "derived_from", turn=mem.clock.now)
        mem.cold.link(s["cold_id"], mid)
        notified = []
        if s["decision"] != "merge" or s.get("recipients"):
            notified = notify.push(mem, mid, agent, s.get("recipients"))
        mem.decisions.append({
            "turn": mem.clock.now, "agent": agent, "event_id": s.get("event_id"),
            "decision": s["decision"], "memory_id": mid, "target": (s.get("best") or {}).get("id"),
            "superseded": s.get("superseded"), "notified": notified,
            "compressed": s["comp_meta"].get("compressed", False),
        })
        return {"notified": notified}

    g = StateGraph(WriteState)
    g.add_node("store_original", store_original)
    g.add_node("compress", compress)
    g.add_node("extract_keywords", extract_keywords)
    g.add_node("score_candidates", score_candidates)
    g.add_node("merge", do_merge)
    g.add_node("update", do_update)
    g.add_node("create", do_create)
    g.add_node("link_notify", link_notify)
    g.add_edge(START, "store_original")
    g.add_edge("store_original", "compress")
    g.add_edge("compress", "extract_keywords")
    g.add_edge("extract_keywords", "score_candidates")
    g.add_conditional_edges("score_candidates", lambda s: s["decision"], ["merge", "update", "create"])
    for n in ("merge", "update", "create"):
        g.add_edge(n, "link_notify")
    g.add_edge("link_notify", END)
    return g.compile()
