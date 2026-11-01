"""Shared graph memory for multi-agent LLM systems (write/read paths, trace, cold storage)."""
from graphmem.config import load_config, Cfg
from graphmem.models import ReadResult

__all__ = ["load_config", "Cfg", "ReadResult", "GraphMemory"]


def __getattr__(name):
    # Lazy import so light modules can be used without pulling in LangGraph.
    if name == "GraphMemory":
        from graphmem.memory import GraphMemory
        return GraphMemory
    raise AttributeError(name)
