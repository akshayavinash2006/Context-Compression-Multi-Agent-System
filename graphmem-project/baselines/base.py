"""MemoryBackend interface shared by GraphMemory and all baselines (plan §12, §15)."""
from __future__ import annotations

from collections import Counter

from graphmem.clock import Clock
from graphmem.models import ReadResult, format_memory_line
from graphmem.tokens import TokenCounter


class MemoryBackend:
    """Flat message store. Subclasses override `select()` to choose what `read()` returns."""

    name = "base"

    def __init__(self, cfg, counter: TokenCounter | None = None):
        self.cfg = cfg
        self.count_tokens = counter or TokenCounter(cfg.agents.get("token_counter", "tiktoken:cl100k_base"))
        self.clock = Clock()
        self.messages: list[dict] = []
        self.by_id: dict[str, dict] = {}
        self.agents: dict[str, dict] = {}
        self.counters: Counter = Counter()
        self.decisions: list[dict] = []
        self.deliveries: list[dict] = []

    # ---------- interface ----------
    def register_agent(self, name: str, role: str, profile_keywords: list[str]) -> None:
        self.agents[name] = {"role": role, "profile": list(profile_keywords)}

    def write(self, agent: str, text: str, recipients=None, derived_from=None, event_id=None, kind=None) -> str:
        mid = f"M{len(self.messages) + 1}"
        msg = {"id": mid, "agent": agent, "turn": self.clock.now, "text": text, "event_id": event_id,
               "derived_from": list(derived_from or []),
               "kind": kind or ("derived" if derived_from else "observation"),
               "tokens": self.count_tokens(text)}
        self.messages.append(msg)
        self.by_id[mid] = msg
        self.counters["writes"] += 1
        self.counters["orig_tokens_written"] += msg["tokens"]
        self.on_write(msg)
        return mid

    def on_write(self, msg: dict) -> None:
        pass

    def select(self, agent: str, query: str, token_budget: int | None) -> list[dict]:
        raise NotImplementedError

    def read(self, agent: str, query: str, token_budget: int | None = None, record: bool = True) -> ReadResult:
        prefix, msgs = self.prefix_text(), self.select(agent, query, token_budget)
        lines = [self.line(m) for m in msgs]
        text = "\n".join(([prefix] if prefix else []) + lines)
        self.counters["reads"] += 1
        return ReadResult(memory_ids=[m["id"] for m in msgs], context_text=text,
                          tokens_used=self.count_tokens(text), scores={})

    def prefix_text(self) -> str:
        return ""

    def line(self, m: dict) -> str:
        return format_memory_line(m["id"], m["agent"], m["turn"], m["text"])

    def after_turn(self, agent: str, delivered: list[str], cited: list[str]) -> None:
        self.deliveries.append({"turn": self.clock.now, "agent": agent,
                                "delivered": list(delivered), "cited": list(cited)})

    def expand(self, memory_id: str) -> list[dict]:
        self.counters["expand"] += 1
        m = self.by_id.get(memory_id)
        return [{"agent": m["agent"], "turn": m["turn"], "text": m["text"], "event_id": m["event_id"]}] if m else []

    def trace(self, memory_id: str, max_depth: int = 6) -> dict:
        """Flat stores keep no provenance graph; the cited IDs are the only link."""
        def node(mid, depth, path):
            m = self.by_id[mid]
            parents = [p for p in m["derived_from"] if p in self.by_id and p not in path and p != mid]
            return {"id": mid, "text": m["text"], "agents": [m["agent"]], "kind": m["kind"], "turn": m["turn"],
                    "originals": [{"agent": m["agent"], "turn": m["turn"], "text": m["text"],
                                   "event_id": m["event_id"]}],
                    "superseded": False, "archived": False, "supersedes": [],
                    "sources": [node(p, depth + 1, path | {mid}) for p in parents] if depth < max_depth else []}
        return node(memory_id, 0, frozenset())

    def maintain(self) -> dict:
        return {"turn": self.clock.now, **self.stats()}

    def tick(self) -> int:
        return self.clock.tick()

    @property
    def now(self) -> int:
        return self.clock.now

    def stats(self) -> dict:
        return {"active_memories": len(self.messages),
                "stored_tokens_active": sum(m["tokens"] for m in self.messages),
                "stored_tokens_all": sum(m["tokens"] for m in self.messages),
                "writes": self.counters["writes"], "expands": self.counters["expand"],
                "orig_tokens_written": self.counters["orig_tokens_written"],
                "stored_tokens_written": self.counters["orig_tokens_written"], "compression_ratio": 1.0}

    def as_retriever(self, agent: str, token_budget: int | None = None):
        from graphmem.lc import MemoryRetriever
        return MemoryRetriever(memory=self, agent=agent, token_budget=token_budget)

    # helper for budgeted baselines
    def fit_budget(self, msgs: list[dict], budget: int) -> list[dict]:
        out, used = [], 0
        for m in msgs:
            n = self.count_tokens(self.line(m)) + 1
            if used + n > budget:
                continue
            out.append(m)
            used += n
        return out
