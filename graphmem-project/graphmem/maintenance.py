"""Maintenance steps: archive, keyword merge, cleanup, stats (plan §11).

Decay needs no step: strength is computed lazily from base_strength and last_used.
"""
from __future__ import annotations

import itertools

from graphmem.keywords import digits
from graphmem.retrieve import strength


def archive(mem) -> int:
    n = 0
    for mid in list(mem.store.active_memories()):
        if strength(mem.store, mem.cfg, mid, mem.clock.now) < mem.cfg.strength.archive_below:
            mem.store.g.nodes[mid]["archived"] = True
            mem.store.deactivate(mid)
            n += 1
    return n


def merge_keywords(mem) -> int:
    """Merge keyword pairs that share a memory neighbour and are near-synonyms."""
    store, cfg = mem.store, mem.cfg
    thr = cfg.maintenance.keyword_merge_cosine
    pairs = set()
    for mid in store.active_memories():
        kids = sorted(store.keywords_of(mid))
        pairs |= set(itertools.combinations(kids, 2))
    merged = 0
    gone: set[str] = set()
    for a, b in sorted(pairs):
        if a in gone or b in gone or a not in store.g or b not in store.g:
            continue
        ca, cb = store.g.nodes[a]["canonical"], store.g.nodes[b]["canonical"]
        if cfg.resolution.digit_guard and digits(ca) != digits(cb):
            continue
        if mem.embedder.cos(ca, cb) < thr:
            continue
        keep, drop = (a, b) if store.g.nodes[a]["df"] >= store.g.nodes[b]["df"] else (b, a)
        _merge_pair(mem, keep, drop)
        gone.add(drop)
        merged += 1
    return merged


def _merge_pair(mem, keep: str, drop: str):
    store = mem.store
    for mid in store.into(drop, "mentions"):
        active = store.is_active(mid)
        if active:
            store.deactivate(mid)
        store.g.remove_edge(mid, drop, key="mentions")
        if not store.has_edge(mid, keep, "mentions"):
            store.add_edge(mid, keep, "mentions", weight=1.0)
        if active:
            store.activate(mid)
    store.g.nodes[keep]["aliases"] |= store.g.nodes[drop]["aliases"]
    mem.index.redirect(drop, keep)
    if drop in store.view:
        store.view.remove_node(drop)
    store.g.remove_node(drop)


def cleanup(mem) -> int:
    store = mem.store
    dead = [k for k, d in store.g.nodes(data=True)
            if d["type"] == "keyword" and d["df"] == 0 and not store.into(k, "mentions")]
    for k in dead:
        mem.index.remove(k)
        store.g.remove_node(k)
    store.refresh_weights()
    return len(dead)


def log_stats(mem) -> dict:
    return {"turn": mem.clock.now, **mem.stats()}
