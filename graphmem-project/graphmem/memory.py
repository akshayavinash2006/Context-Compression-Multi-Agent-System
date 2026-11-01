"""GraphMemory: public API (plan §12). write/read/maintain invoke compiled LangGraph graphs."""
from __future__ import annotations

from collections import Counter

import networkx as nx
import numpy as np
import yaml

from graphmem import notify, trace as trace_mod
from graphmem.clock import Clock
from graphmem.cold_store import ColdStore
from graphmem.compress import Compressor
from graphmem.config import Cfg, load_config, resolve_path
from graphmem.embed import Embedder
from graphmem.graph_store import GraphStore
from graphmem.graphs import build_maintain_graph, build_read_graph, build_write_graph
from graphmem.keywords import KeywordExtractor
from graphmem.models import ReadResult, format_memory_line
from graphmem.nli import make_nli
from graphmem.resolve import KeywordIndex
from graphmem.runtime import set_heavy_mode
from graphmem.tokens import TokenCounter


class SharedModels:
    """Heavy models loaded once and shared by every GraphMemory instance in a process."""
    _cache: dict = {}

    @classmethod
    def get(cls, cfg: Cfg):
        key = str(cfg.runtime.get("use_heavy_models", "auto"))
        if key not in cls._cache:
            set_heavy_mode(cfg.runtime.get("use_heavy_models", "auto"))
            embedder = Embedder(cfg)
            with open(resolve_path(cfg.keywords.get("vocabulary", "domain/vocabulary.yaml")), encoding="utf-8") as f:
                antonyms = (yaml.safe_load(f) or {}).get("antonyms", [])
            counter = TokenCounter(cfg.agents.get("token_counter", "tiktoken:cl100k_base"))
            cls._cache[key] = {
                "embedder": embedder,
                "counter": counter,
                "extractor": KeywordExtractor(cfg, embedder),
                "nli": make_nli(cfg, antonyms),
                "compressor": Compressor(cfg.compression, counter),
            }
        return cls._cache[key]


def apply_embedding_overrides(cfg: Cfg, embedder_kind: str) -> Cfg:
    if embedder_kind == "hash":
        for k, v in (cfg.get("hash_embedding_overrides") or {}).items():
            cfg.set_path(k, v)
    return cfg


class GraphMemory:
    name = "graph"

    def __init__(self, cfg: Cfg | None = None, db_path: str = ":memory:", scenario_id: str | None = None,
                 compressor=None):
        cfg = (cfg or load_config()).copy()
        shared = SharedModels.get(cfg)
        self.embedder: Embedder = shared["embedder"]
        self.cfg = apply_embedding_overrides(cfg, self.embedder.kind)
        self.count_tokens: TokenCounter = shared["counter"]
        self.extractor: KeywordExtractor = shared["extractor"]
        self.nli = shared["nli"]
        if compressor is not None:
            self.compressor = compressor
        elif self.cfg.compression.rate != shared["compressor"].cfg.rate:
            self.compressor = Compressor(self.cfg.compression, self.count_tokens)
        else:
            self.compressor = shared["compressor"]
        self.clock = Clock()
        self.store = GraphStore()
        self.cold = ColdStore(db_path, scenario_id)
        self.index = KeywordIndex(self.cfg, self.embedder, self.store)
        self.mem_vecs: dict[str, np.ndarray] = {}
        self.counters: Counter = Counter()
        self.decisions: list[dict] = []
        self.deliveries: list[dict] = []
        self._next_id = 1
        self._write_graph = build_write_graph(self)
        self._read_graph = build_read_graph(self)
        self._maintain_graph = build_maintain_graph(self)

    # ---------- helpers used by graph nodes ----------
    def embed(self, text: str) -> np.ndarray:
        return self.embedder(text)

    def memory_vec(self, mid: str) -> np.ndarray:
        if mid not in self.mem_vecs:
            self.mem_vecs[mid] = self.embedder(self.store.g.nodes[mid]["text"])
        return self.mem_vecs[mid]

    def format_line(self, mid: str) -> str:
        d = self.store.g.nodes[mid]
        writer = (self.store.writers(mid) or ["?"])[0]
        return format_memory_line(mid, writer, d["created_at"], d["text"])

    def _create_memory(self, s: dict) -> str:
        mid = f"M{self._next_id}"
        self._next_id += 1
        now = self.clock.now
        meta = s["comp_meta"]
        kind = s.get("kind") or ("derived" if s.get("derived_from") else "observation")
        self.store.add_memory(
            mid, text=s["compressed"], tokens=self.count_tokens(s["compressed"]),
            orig_tokens=meta.get("orig_tokens", self.count_tokens(s["text"])), cold_id=s["cold_id"],
            kind=kind, created_at=now, last_used=now, use_count=0,
            base_strength=self.cfg.strength.initial, valid_from=now, valid_until=None,
            archived=False, n_writers=0)
        for kid in s["keyword_ids"]:
            self.store.add_edge(mid, kid, "mentions", weight=1.0)
        self.store.activate(mid)
        self.mem_vecs[mid] = self.embedder(s["compressed"])
        self.counters["orig_tokens_written"] += meta.get("orig_tokens", 0)
        self.counters["stored_tokens_written"] += self.store.g.nodes[mid]["tokens"]
        return mid

    def _record_delivery(self, agent: str, ids: list[str]):
        now = self.clock.now
        if f"A:{agent}" not in self.store.g:
            self.store.add_agent(agent, agent, {})
        a = self.store.agent(agent)
        for mid in ids:
            d = self.store.g.nodes[mid]
            from graphmem.retrieve import strength
            s_now = strength(self.store, self.cfg, mid, now)
            d["base_strength"] = min(self.cfg.strength.max, s_now + self.cfg.strength.read_boost)
            d["last_used"] = now
            d["use_count"] += 1
            self.store.add_edge(mid, f"A:{agent}", "read_by", turn=now)
            if mid in a["inbox"]:
                a["inbox"].remove(mid)

    # ---------- core API ----------
    def register_agent(self, name: str, role: str, profile_keywords: list[str]) -> None:
        profile = {self.extractor.normalize(k): 1.0 for k in profile_keywords}
        self.store.add_agent(name, role, profile)

    def write(self, agent: str, text: str, recipients: list[str] | None = None,
              derived_from: list[str] | None = None, event_id: str | None = None,
              kind: str | None = None) -> str:
        self.counters["writes"] += 1
        out = self._write_graph.invoke({
            "agent": agent, "text": text, "recipients": recipients or [],
            "derived_from": list(derived_from or []), "event_id": event_id, "kind": kind})
        return out["memory_id"]

    def read(self, agent: str, query: str, token_budget: int | None = None, record: bool = True) -> ReadResult:
        budget = token_budget or self.cfg.retrieval.token_budget_default
        out = self._read_graph.invoke({"agent": agent, "query": query, "budget": budget, "record": record})
        res = ReadResult(memory_ids=out["memory_ids"], context_text=out["context_text"],
                         tokens_used=out["tokens_used"], scores=out["scores"],
                         seeds_used={k: round(v, 4) for k, v in out["seeds"].items()},
                         fallback=out.get("fallback"))
        self.counters["reads"] += 1
        return res

    def after_turn(self, agent: str, delivered: list[str], cited: list[str]) -> None:
        """R4 second half: profile learning and unnecessary-context bookkeeping."""
        notify.learn_profile(self, agent, cited)
        self.deliveries.append({"turn": self.clock.now, "agent": agent,
                                "delivered": list(delivered), "cited": list(cited)})

    def expand(self, memory_id: str) -> list[dict]:
        self.counters["expand"] += 1
        return self.cold.get(memory_id) if memory_id in self.store.g else []

    def trace(self, memory_id: str, max_depth: int = 6) -> dict:
        return trace_mod.trace(self.store, self.cold, memory_id, max_depth)

    # ---------- housekeeping ----------
    def maintain(self) -> dict:
        out = self._maintain_graph.invoke({})
        return {"archived": out["archived"], "keywords_merged": out["keywords_merged"],
                "keywords_removed": out["keywords_removed"], **out["stats"]}

    def tick(self) -> int:
        return self.clock.tick()

    @property
    def now(self) -> int:
        return self.clock.now

    def stats(self) -> dict:
        s = self.store.stats()
        c = self.counters
        s.update({
            "writes": c["writes"], "merges": c["merge"], "updates": c["update"], "creates": c["create"],
            "compress_fallbacks": c["compress_fallbacks"], "expands": c["expand"],
            "orig_tokens_written": c["orig_tokens_written"],
            "stored_tokens_written": c["stored_tokens_written"],
            "compression_ratio": round(c["orig_tokens_written"] / c["stored_tokens_written"], 3)
            if c["stored_tokens_written"] else 1.0,
            "embedder": self.embedder.kind, "nli": self.nli.kind,
            "compressor": getattr(self.compressor, "backend_name", "custom"),
        })
        return s

    def export_graph(self, path: str):
        """GraphML for Gephi: attributes flattened to str/int/float."""
        h = nx.MultiDiGraph()
        for n, d in self.store.g.nodes(data=True):
            h.add_node(n, **{k: _flat(v) for k, v in d.items()})
        for u, v, k, d in self.store.g.edges(keys=True, data=True):
            h.add_edge(u, v, key=k, label=k, **{a: _flat(b) for a, b in d.items()})
        nx.write_graphml(h, path)

    def as_retriever(self, agent: str, token_budget: int | None = None):
        from graphmem.lc import MemoryRetriever
        return MemoryRetriever(memory=self, agent=agent, token_budget=token_budget)


def _flat(v):
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return v
    return str(sorted(v) if isinstance(v, set) else v)
