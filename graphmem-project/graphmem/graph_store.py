"""Typed NetworkX graph + undirected retrieval view (plan §7)."""
from __future__ import annotations

import math

import networkx as nx


class GraphStore:
    def __init__(self):
        self.g = nx.MultiDiGraph()
        self.view = nx.Graph()          # active memories + their keywords, edge weight = idf

    # ---------- nodes ----------
    def add_agent(self, name: str, role: str, profile: dict[str, float]):
        aid = f"A:{name}"
        if aid in self.g:
            self.g.nodes[aid].update(role=role, profile=dict(profile))
        else:
            self.g.add_node(aid, type="agent", name=name, role=role, profile=dict(profile), inbox=[])
        return aid

    def agent(self, name: str) -> dict:
        return self.g.nodes[f"A:{name}"]

    def agents(self) -> list[str]:
        return [self.g.nodes[n]["name"] for n, d in self.g.nodes(data=True) if d["type"] == "agent"]

    def add_keyword(self, kid: str, canonical: str, kind: str, alias: str):
        if kid not in self.g:
            self.g.add_node(kid, type="keyword", canonical=canonical, aliases={alias.lower(), canonical},
                            kind=kind, df=0)

    def add_memory(self, mid: str, **attrs):
        self.g.add_node(mid, type="memory", **attrs)

    def is_active(self, mid: str) -> bool:
        d = self.g.nodes[mid]
        return d.get("type") == "memory" and not d["archived"] and d["valid_until"] is None

    # ---------- edges ----------
    def add_edge(self, u: str, v: str, key: str, **attrs):
        self.g.add_edge(u, v, key=key, **attrs)

    def has_edge(self, u: str, v: str, key: str) -> bool:
        return self.g.has_edge(u, v, key=key)

    def out(self, u: str, key: str) -> list[str]:
        return [v for _, v, k in self.g.out_edges(u, keys=True) if k == key]

    def into(self, v: str, key: str) -> list[str]:
        return [u for u, _, k in self.g.in_edges(v, keys=True) if k == key]

    def keywords_of(self, mid: str) -> list[str]:
        return self.out(mid, "mentions")

    def memories_of(self, kid: str, active_only: bool = True) -> list[str]:
        ms = self.into(kid, "mentions")
        return [m for m in ms if self.is_active(m)] if active_only else ms

    def writers(self, mid: str) -> list[str]:
        return [v.split(":", 1)[1] for v in self.out(mid, "written_by")]

    # ---------- activation / retrieval view ----------
    def activate(self, mid: str):
        """Put a memory and its keywords into the retrieval view; df += 1."""
        self.view.add_node(mid, type="memory")
        for kid in self.keywords_of(mid):
            self.g.nodes[kid]["df"] += 1
            self.view.add_node(kid, type="keyword")
            self.view.add_edge(mid, kid, weight=self.idf(kid))

    def deactivate(self, mid: str):
        """Remove from the retrieval view (superseded or archived); df -= 1."""
        if mid in self.view:
            self.view.remove_node(mid)
        for kid in self.keywords_of(mid):
            d = self.g.nodes[kid]
            d["df"] = max(0, d["df"] - 1)
            if d["df"] == 0 and kid in self.view:
                self.view.remove_node(kid)

    def active_memories(self) -> list[str]:
        return [n for n, d in self.view.nodes(data=True) if d["type"] == "memory"]

    def n_active(self) -> int:
        return sum(1 for _, d in self.view.nodes(data=True) if d["type"] == "memory")

    def idf(self, kid: str) -> float:
        n = self.n_active()
        df = self.g.nodes[kid]["df"] if kid in self.g else 0
        return math.log((n + 1) / (df + 1)) + 1.0

    def refresh_weights(self):
        for u, v, d in self.view.edges(data=True):
            kid = u if u.startswith("K:") else v
            d["weight"] = self.idf(kid)

    # ---------- stats ----------
    def stats(self) -> dict:
        nodes = {"agent": 0, "memory": 0, "keyword": 0}
        for _, d in self.g.nodes(data=True):
            nodes[d["type"]] = nodes.get(d["type"], 0) + 1
        edges: dict[str, int] = {}
        for _, _, k in self.g.edges(keys=True):
            edges[k] = edges.get(k, 0) + 1
        active = self.active_memories()
        return {
            "nodes": nodes, "edges": edges,
            "active_memories": len(active),
            "archived_memories": sum(1 for _, d in self.g.nodes(data=True)
                                     if d["type"] == "memory" and d["archived"]),
            "superseded_memories": sum(1 for _, d in self.g.nodes(data=True)
                                       if d["type"] == "memory" and d["valid_until"] is not None),
            "stored_tokens_active": sum(self.g.nodes[m]["tokens"] for m in active),
            "stored_tokens_all": sum(d["tokens"] for _, d in self.g.nodes(data=True) if d["type"] == "memory"),
        }
