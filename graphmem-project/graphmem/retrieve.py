"""R1-R3: seed matching, Personalized PageRank ranking, MMR + token budget (plan §9)."""
from __future__ import annotations

import math

import networkx as nx
import numpy as np


def strength(store, cfg, mid: str, now: int) -> float:
    d = store.g.nodes[mid]
    return d["base_strength"] * math.exp(-cfg.strength.decay_lambda * max(0, now - d["last_used"]))


# ---------- R1 ----------
def match_query(mem, agent: str, query: str) -> tuple[dict[str, float], list[str]]:
    """Seeds from query keywords (exact/fuzzy/embedding, no creation), profile and inbox.

    Each seed group's weights sum to its configured weight (query 1.0, profile 0.2,
    inbox 0.5), so a long profile or a full inbox can never outweigh the query itself.
    """
    cfg, store = mem.cfg, mem.store
    r = cfg.retrieval
    q: dict[str, float] = {}
    terms = mem.extractor.extract(query, with_phrases=True)
    for t in terms:
        kid, conf = mem.index.resolve(t.canonical, t.kind, create=False)
        if kid is not None and kid in store.view:
            q[kid] = max(q.get(kid, 0.0), conf)
        else:
            for kid2, cos in mem.index.nearest(t.canonical, r.fallback_embed_top_k, r.fallback_embed_min_cos):
                if kid2 in store.view:
                    q[kid2] = max(q.get(kid2, 0.0), cos)
        for rel in mem.extractor.related.get(t.canonical, []):        # vocabulary relations
            kid3 = mem.index.lookup(rel)
            if kid3 and kid3 in store.view:
                q[kid3] = max(q.get(kid3, 0.0), 0.7)
    prof: dict[str, float] = {}
    inbox: dict[str, float] = {}
    if f"A:{agent}" in store.g:
        a = store.agent(agent)
        for canon, w in a["profile"].items():
            kid = mem.index.lookup(canon)
            if kid and kid in store.view:
                prof[kid] = prof.get(kid, 0.0) + w
        for mid in a["inbox"]:
            if mid in store.view:
                inbox[mid] = 1.0
    seeds: dict[str, float] = {}
    for group, weight in ((q, r.seed_weight_query), (prof, r.seed_weight_profile), (inbox, r.seed_weight_inbox)):
        total = sum(group.values())
        for k, v in group.items():
            seeds[k] = seeds.get(k, 0.0) + weight * v / total
    return seeds, list(q)


# ---------- R2 ----------
def ppr_rank(store, seeds: dict[str, float], cfg, now: int) -> list[tuple[str, float]]:
    view = store.view
    nodes: set[str] = set()
    for s in seeds:
        if s in view:
            nodes |= set(nx.single_source_shortest_path_length(view, s, cutoff=cfg.retrieval.ego_radius))
    if not nodes:
        return []
    sub = view.subgraph(nodes).copy()
    for u, v, d in sub.edges(data=True):          # current IDF at retrieval time
        d["weight"] = store.idf(u if u.startswith("K:") else v)
    pers = {n: w for n, w in seeds.items() if n in sub and w > 0}
    if not pers:
        return []
    try:
        pr = nx.pagerank(sub, alpha=cfg.retrieval.ppr_alpha, personalization=pers, weight="weight")
    except nx.PowerIterationFailedConvergence:
        pr = nx.pagerank(sub, alpha=cfg.retrieval.ppr_alpha, personalization=pers, weight="weight",
                         max_iter=1000, tol=1e-4)
    use_s = cfg.retrieval.get("use_strength", True)
    scored = [(m, pr[m] * (strength(store, cfg, m, now) if use_s else 1.0))
              for m in sub if sub.nodes[m]["type"] == "memory"]
    return sorted(scored, key=lambda x: (-x[1], x[0]))


def keyword_count_rank(store, seeds: dict[str, float], cfg, now: int) -> list[tuple[str, float]]:
    """Ablation 'no PPR': number of matched keywords x strength."""
    counts: dict[str, float] = {}
    for s, w in seeds.items():
        if s.startswith("K:") and s in store.view:
            for m in store.view.neighbors(s):
                counts[m] = counts.get(m, 0.0) + w
        elif s in store.view:
            counts[s] = counts.get(s, 0.0) + w
    use_s = cfg.retrieval.get("use_strength", True)
    scored = [(m, c * (strength(store, cfg, m, now) if use_s else 1.0)) for m, c in counts.items()]
    return sorted(scored, key=lambda x: (-x[1], x[0]))


def vector_rank(mem, query: str) -> list[tuple[str, float]]:
    active = mem.store.active_memories()
    if not active:
        return []
    q = mem.embed(query)
    sims = [(m, float(q @ mem.memory_vec(m))) for m in active]
    sims.sort(key=lambda x: (-x[1], x[0]))
    return sims[: mem.cfg.retrieval.vector_fallback_top_k]


# ---------- R3 ----------
def eligible(store, cfg, agent: str, mid: str, now: int = 0) -> bool:
    d = store.g.nodes[mid]
    if d["archived"]:
        return False
    if d["valid_until"] is not None and not cfg.retrieval.include_superseded:
        return False
    aid = f"A:{agent}"
    if cfg.retrieval.skip_already_read and store.has_edge(mid, aid, "read_by"):
        # agents are stateless between calls, so only skip what was sent recently
        window = cfg.retrieval.get("skip_read_window")
        last = store.g.edges[mid, aid, "read_by"]["turn"]       # re-adding the key updates the turn
        if window is None or now - last <= window:
            return False
    return True


def mmr_select(mem, ranked: list[tuple[str, float]], lam: float) -> list[tuple[str, float]]:
    if not ranked:
        return []
    pool = ranked[: mem.cfg.retrieval.get("mmr_pool", 50)]
    top = max(s for _, s in pool) or 1.0
    vecs = {m: mem.memory_vec(m) for m, _ in pool}
    selected: list[tuple[str, float]] = []
    remaining = list(pool)
    while remaining:
        best, best_val = None, -np.inf
        for m, s in remaining:
            red = max((float(vecs[m] @ vecs[x]) for x, _ in selected), default=0.0)
            val = lam * (s / top) - (1 - lam) * red
            if val > best_val:
                best, best_val = (m, s), val
        selected.append(best)
        remaining.remove(best)
    return selected


def fill_budget(mem, agent: str, ranked: list[tuple[str, float]], budget: int):
    store, cfg = mem.store, mem.cfg
    cands = [(m, s) for m, s in ranked if eligible(store, cfg, agent, m, mem.clock.now)]
    ordered = mmr_select(mem, cands, cfg.retrieval.mmr_lambda)
    ids, lines, scores, used = [], [], {}, 0
    for m, s in ordered:
        line = mem.format_line(m)
        n = mem.count_tokens(line) + 1
        if used + n > budget:
            continue
        ids.append(m)
        lines.append(line)
        scores[m] = round(float(s), 6)
        used += n
    return ids, "\n".join(lines), used, scores
