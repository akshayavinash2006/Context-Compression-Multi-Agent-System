"""Read path R1-R4 as a LangGraph StateGraph (plan §9).

START -> match_keywords --(seeds)----> rank ------------> fill_budget -> record_read -> END
                        \--(no seeds)-> vector_fallback -/
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from graphmem import retrieve


class ReadState(TypedDict, total=False):
    agent: str
    query: str
    budget: int
    record: bool
    seeds: dict
    query_keywords: list[str]
    ranked: list
    fallback: str | None
    memory_ids: list[str]
    context_text: str
    tokens_used: int
    scores: dict


def build_read_graph(mem):
    def match_keywords(s: ReadState):                              # R1
        seeds, qk = retrieve.match_query(mem, s["agent"], s["query"])
        return {"seeds": seeds, "query_keywords": qk}

    def route(s: ReadState):
        return "rank" if s["seeds"] else "vector_fallback"

    def rank(s: ReadState):                                        # R2
        ranker = retrieve.keyword_count_rank if mem.cfg.retrieval.get("ranker") == "keyword_count" \
            else retrieve.ppr_rank
        ranked = ranker(mem.store, s["seeds"], mem.cfg, mem.clock.now)
        return {"ranked": ranked, "fallback": None}

    def vector_fallback(s: ReadState):
        return {"ranked": retrieve.vector_rank(mem, s["query"]), "fallback": "vector"}

    def fill_budget(s: ReadState):                                 # R3
        ids, text, used, scores = retrieve.fill_budget(mem, s["agent"], s["ranked"], s["budget"])
        return {"memory_ids": ids, "context_text": text, "tokens_used": used, "scores": scores,
                "fallback": s.get("fallback") if ids else (s.get("fallback") or "empty")}

    def record_read(s: ReadState):                                 # R4 (delivery part)
        if s.get("record", True):
            mem._record_delivery(s["agent"], s["memory_ids"])
        return {}

    g = StateGraph(ReadState)
    g.add_node("match_keywords", match_keywords)
    g.add_node("rank", rank)
    g.add_node("vector_fallback", vector_fallback)
    g.add_node("fill_budget", fill_budget)
    g.add_node("record_read", record_read)
    g.add_edge(START, "match_keywords")
    g.add_conditional_edges("match_keywords", route, ["rank", "vector_fallback"])
    g.add_edge("rank", "fill_budget")
    g.add_edge("vector_fallback", "fill_budget")
    g.add_edge("fill_budget", "record_read")
    g.add_edge("record_read", END)
    return g.compile()
