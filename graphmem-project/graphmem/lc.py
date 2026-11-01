"""LangChain integration: retriever over any memory backend, and memory tools for agents."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool


class MemoryRetriever(BaseRetriever):
    """Exposes `backend.read()` as a LangChain retriever (one Document per delivered memory)."""

    memory: Any
    agent: str
    token_budget: int | None = None
    record: bool = False

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        res = self.memory.read(self.agent, query, token_budget=self.token_budget, record=self.record)
        lines = res.context_text.splitlines()
        docs = []
        for mid, line in zip(res.memory_ids, lines):
            docs.append(Document(page_content=line, metadata={"id": mid, "score": res.scores.get(mid)}))
        return docs


GraphMemoryRetriever = MemoryRetriever


def _expand_text(memory, memory_id: str) -> str:
    memory_id = memory_id.strip().strip("[]")
    originals = memory.expand(memory_id)
    if not originals:
        return f"No original text stored for {memory_id}."
    return "\n".join(f"[{memory_id} original | {o['agent']} | t={o['turn']}] {o['text']}" for o in originals)


def _trace_text(memory, memory_id: str) -> str:
    memory_id = memory_id.strip().strip("[]")
    try:
        tr = memory.trace(memory_id)
    except Exception:
        return f"No provenance available for {memory_id}."

    def strip(n):
        return {"id": n["id"], "agents": n["agents"], "text": n["text"],
                "sources": [strip(s) for s in n.get("sources", [])]}
    return json.dumps(strip(tr))[:4000]


@tool
def expand_memory(memory_id: str, config: RunnableConfig) -> str:
    """Return the full original text(s) of a memory such as 'M7', when its compressed text looks incomplete."""
    return _expand_text(config["configurable"]["memory"], memory_id)


@tool
def trace_memory(memory_id: str, config: RunnableConfig) -> str:
    """Return the provenance of a memory such as 'M4': the facts and agents it was derived from."""
    return _trace_text(config["configurable"]["memory"], memory_id)


# Tools used by the agent subgraph: the memory backend is injected from config["configurable"].
MEMORY_TOOLS = [expand_memory, trace_memory]


def make_memory_tools(memory):
    """Same tools bound to one backend, for use outside LangGraph."""
    @tool("expand_memory")
    def _expand(memory_id: str) -> str:
        """Return the full original text(s) of a memory such as 'M7'."""
        return _expand_text(memory, memory_id)

    @tool("trace_memory")
    def _trace(memory_id: str) -> str:
        """Return the provenance of a memory such as 'M4'."""
        return _trace_text(memory, memory_id)
    return [_expand, _trace]
