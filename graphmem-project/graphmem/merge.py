"""W3: merge-or-create scoring and decision rule (plan §8.3)."""
from __future__ import annotations


def weighted_jaccard(a: set[str], b: set[str], idf) -> float:
    union = a | b
    if not union:
        return 0.0
    den = sum(idf(k) for k in union)
    return sum(idf(k) for k in a & b) / den if den else 0.0


def decide(j: float, cos: float, p_contra: float, cfg) -> str:
    m = cfg.merge
    if j >= m.jaccard_merge and cos >= m.cosine_merge and p_contra < m.nli_contradiction_block_merge:
        return "merge"
    if (j >= m.jaccard_merge or cos >= m.cosine_update) and p_contra >= m.nli_contradiction_update:
        return "update"
    return "create"


def jaccard_set(store, kids, kinds) -> set[str]:
    """Domain keywords of a memory (IDs, vocabulary, NER); all keywords if it has none."""
    dom = {k for k in kids if store.g.nodes[k]["kind"] in kinds}
    return dom or set(kids)


ID_KINDS = {"machine", "po"}


def id_keywords(store, kids) -> set[str]:
    return {k for k in kids if store.g.nodes[k]["kind"] in ID_KINDS}


def score_candidates(mem, new_text: str, new_kids: list[str],
                     exclude: list[str] | None = None) -> tuple[str, dict | None, list[dict]]:
    """Returns (decision, best candidate, all scored candidates).

    Guards (in addition to the digit guard on keywords):
    - ID guard: two memories about different machines / POs are never merged or updated.
    - Sources: a memory is never merged into, or allowed to supersede, the memories it was
      derived from (an answer builds on its facts, it does not replace them).
    """
    cfg, store = mem.cfg, mem.store
    if not cfg.merge.get("enabled", True):
        return "create", None, []
    kinds = set(cfg.merge.jaccard_kinds)
    new_set = jaccard_set(store, new_kids, kinds)
    new_ids = id_keywords(store, new_kids)
    skip = set(exclude or [])
    shared: dict[str, int] = {}
    for kid in new_kids:
        for m in store.memories_of(kid):
            shared[m] = shared.get(m, 0) + 1
    cands = [m for m, n in shared.items() if n >= cfg.merge.candidate_min_shared_keywords
             and m not in skip and id_keywords(store, store.keywords_of(m)) == new_ids]
    if not cands:
        return "create", None, []
    new_vec = mem.embed(new_text)
    scored = []
    for m in cands:
        j = weighted_jaccard(new_set, jaccard_set(store, store.keywords_of(m), kinds), store.idf)
        cos = float(new_vec @ mem.memory_vec(m))
        if j < cfg.merge.jaccard_skip_below and cos < cfg.merge.cosine_update:
            continue
        scored.append({"id": m, "jaccard": round(j, 4), "cos": round(cos, 4),
                       "combined": 0.5 * j + 0.5 * cos})
    if not scored:
        return "create", None, []
    scored.sort(key=lambda c: c["combined"], reverse=True)
    top = scored[: cfg.merge.get("top_candidates_nli", 3)]
    for c in top:
        c["p_contra"] = round(mem.nli.contradiction(new_text, store.g.nodes[c["id"]]["text"]), 4)
        c["decision"] = decide(c["jaccard"], c["cos"], c["p_contra"], cfg)
    for wanted in ("merge", "update"):
        for c in top:
            if c["decision"] == wanted:
                return wanted, c, scored
    return "create", top[0], scored
