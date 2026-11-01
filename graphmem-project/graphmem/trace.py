"""trace(): answer -> facts -> agents -> original texts (plan §10.2)."""
from __future__ import annotations


def trace(store, cold, mid: str, max_depth: int = 6) -> dict:
    def node(m: str, depth: int, path: frozenset) -> dict:
        d = store.g.nodes[m]
        parents = [p for p in store.out(m, "derived_from") if p != m and p not in path]
        return {
            "id": m, "text": d["text"], "agents": store.writers(m), "kind": d["kind"],
            "turn": d["created_at"],
            "originals": cold.get(m),
            "superseded": d["valid_until"] is not None,
            "archived": d["archived"],
            "supersedes": store.out(m, "supersedes"),
            "sources": [node(p, depth + 1, path | {m}) for p in parents] if depth < max_depth else [],
        }
    return node(mid, 0, frozenset())


def iter_nodes(tr: dict):
    yield tr
    for s in tr["sources"]:
        yield from iter_nodes(s)


def leaf_event_ids(tr: dict) -> set[str]:
    """Event IDs of the observation facts the answer rests on (for trace precision/recall)."""
    out = set()
    for n in iter_nodes(tr):
        if n["kind"] == "observation":
            out |= {o["event_id"] for o in n["originals"] if o.get("event_id")}
    return out


def trace_size(tr: dict) -> int:
    return sum(1 for _ in iter_nodes(tr))
