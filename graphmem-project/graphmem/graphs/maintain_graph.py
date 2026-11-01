"""Maintenance job as a LangGraph StateGraph (plan §11).

START -> archive -> merge_keywords -> cleanup -> log_stats -> END
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from graphmem import maintenance


class MaintainState(TypedDict, total=False):
    archived: int
    keywords_merged: int
    keywords_removed: int
    stats: dict


def build_maintain_graph(mem):
    g = StateGraph(MaintainState)
    g.add_node("archive", lambda s: {"archived": maintenance.archive(mem)})
    g.add_node("merge_keywords", lambda s: {"keywords_merged": maintenance.merge_keywords(mem)})
    g.add_node("cleanup", lambda s: {"keywords_removed": maintenance.cleanup(mem)})
    g.add_node("log_stats", lambda s: {"stats": maintenance.log_stats(mem)})
    g.add_edge(START, "archive")
    g.add_edge("archive", "merge_keywords")
    g.add_edge("merge_keywords", "cleanup")
    g.add_edge("cleanup", "log_stats")
    g.add_edge("log_stats", END)
    return g.compile()
