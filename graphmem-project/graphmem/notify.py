"""W4: need profiles and push notification (plan §8.4)."""
from __future__ import annotations


def profile_match(store, mid: str, profile: dict[str, float]) -> float:
    kids = store.keywords_of(mid)
    if not kids or not profile:
        return 0.0
    num = den = 0.0
    for kid in kids:
        w = store.idf(kid)
        den += w
        canon = store.g.nodes[kid]["canonical"]
        aliases = store.g.nodes[kid]["aliases"]
        weight = profile.get(canon, 0.0) or max((profile.get(a, 0.0) for a in aliases), default=0.0)
        num += weight * w
    return num / den if den else 0.0


def push(mem, mid: str, writer: str, recipients: list[str] | None = None) -> list[str]:
    cfg, store = mem.cfg, mem.store
    notified = []
    for name in store.agents():
        if name == writer:
            continue
        agent = store.agent(name)
        forced = bool(recipients) and name in recipients
        if not forced:
            if not cfg.notify.enabled:
                continue
            if profile_match(store, mid, agent["profile"]) < cfg.notify.profile_match_threshold:
                continue
        if mid not in agent["inbox"]:
            agent["inbox"].append(mid)
        store.add_edge(mid, f"A:{name}", "sent_to", turn=mem.clock.now)
        notified.append(name)
    return notified


def learn_profile(mem, agent: str, cited: list[str]):
    """Add keywords of cited memories to the agent's profile (weight +0.1, cap 1.0)."""
    cfg, store = mem.cfg, mem.store
    if not cfg.notify.profile_learning or f"A:{agent}" not in store.g:
        return
    prof = store.agent(agent)["profile"]
    inc = cfg.notify.get("profile_learning_weight", 0.1)
    for mid in cited:
        if mid not in store.g:
            continue
        for kid in store.keywords_of(mid):
            c = store.g.nodes[kid]["canonical"]
            prof[c] = min(1.0, prof.get(c, 0.0) + inc)
    if len(prof) > cfg.notify.profile_max_size:
        keep = sorted(prof.items(), key=lambda kv: kv[1], reverse=True)[: cfg.notify.profile_max_size]
        prof.clear()
        prof.update(keep)
