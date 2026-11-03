"""Agent subgraph + orchestrator StateGraph (plan §13.5).

Agent subgraph:  read_context -> call_model -(tool call)-> tools (ToolNode) -> call_model
                                           \-(answer)----> parse_citations -> write_answer
Orchestrator:    next_event -> replay_write | sensor | history | diagnosis -> report
                 -> maintenance? -> next_event ... -> END

Runtime objects (memory backend, model chains, call logger) travel in
config["configurable"], so one compiled graph serves every experimental condition.
"""
from __future__ import annotations

import operator
import re
import sqlite3
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from agents.prompts import (AGENT_PROMPT, DIAGNOSIS_TASK, HISTORY_TASK, REPORT_TASK, ROLES,
                            SENSOR_TASK, role_for, task_kind)
from agents.schemas import ReportSchema, parse_report_text
from graphmem.config import resolve_path
from graphmem.lc import MEMORY_TOOLS
from graphmem.models import format_memory_line, normalize_machine

CITE = re.compile(r"\[(M\d+)\]")
AGENT_FOR_EVENT = {"sensor_reading": "sensor", "operator_note": "sensor", "maintenance": "history",
                   "spec": "history", "question": "diagnosis"}


def _rt(config: RunnableConfig) -> dict:
    return config["configurable"]


THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
THINK_OPEN = re.compile(r"<think>.*", re.S | re.I)          # unclosed block (answer cut off)


def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> blocks that reasoning models (e.g. qwen3) put in their answer."""
    return THINK_OPEN.sub("", THINK.sub("", text)).strip()


def _text(content: Any) -> str:
    if isinstance(content, list):
        content = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content
                           if not (isinstance(p, dict) and p.get("type") in ("thinking", "reasoning")))
    return strip_reasoning(str(content or ""))


# ======================= agent subgraph =======================
class AgentState(TypedDict, total=False):
    agent: str
    event: dict
    task: str
    handoff: dict | None            # Diagnosis output passed to Report: {"memory_id", "text", "turn"}
    context: dict
    delivered: list[str]
    messages: Annotated[list[AnyMessage], add_messages]
    n_tool_calls: int
    answer: str
    report: dict | None
    cited: list[str]
    citation_missing: bool
    memory_id: str
    call_meta: dict


def build_agent_graph():
    def read_context(s: AgentState, config: RunnableConfig):
        rt = _rt(config)
        mem, cfg, agent = rt["memory"], rt["cfg"], s["agent"]
        ev = s["event"]
        query = ev["text"]
        ctx = mem.read(agent, query, token_budget=rt.get("token_budget") or cfg.retrieval.token_budget_default)
        context_text, delivered = ctx.context_text, list(ctx.memory_ids)
        h = s.get("handoff")
        if h:                                       # explicit hand-off, same for every condition
            line = format_memory_line(h["memory_id"], "diagnosis", h["turn"], h["text"])
            context_text = line + ("\n" + context_text if context_text else "")
            delivered = [h["memory_id"]] + [m for m in delivered if m != h["memory_id"]]
        task = {"sensor": SENSOR_TASK.format(text=query), "history": HISTORY_TASK.format(text=query),
                "diagnosis": DIAGNOSIS_TASK.format(question=query),
                "report": REPORT_TASK.format(question=query)}[task_kind(agent)]
        role = role_for(agent, rt.get("zones"))
        msgs = AGENT_PROMPT.format_messages(role=role["role"], role_description=role["description"],
                                            context=context_text or "(no facts available)", task=task)
        rt["logger"].set_context(turn=mem.now, agent=agent, event_id=ev.get("id"))
        return {"context": ctx.to_dict(), "delivered": delivered, "messages": msgs, "task": task,
                "n_tool_calls": 0}

    def call_model(s: AgentState, config: RunnableConfig):
        rt = _rt(config)
        chain = rt["models"].get(s["agent"]) or rt["models"][task_kind(s["agent"])]   # zone agents use the sensor model
        cap = rt["cfg"].agents.max_expand_per_turn
        cb = {"callbacks": [rt["logger"]]}
        if s["agent"] == "report":
            try:
                out, meta = chain.invoke(s["messages"], schema=ReportSchema, config=cb)
                report = out if isinstance(out, ReportSchema) else ReportSchema(**dict(out))
            except Exception:
                out, meta = chain.invoke(s["messages"], config=cb)
                report = parse_report_text(_text(out.content)) or ReportSchema(
                    machine="", root_cause="none", action="", evidence=[])
            report.machine = normalize_machine(report.machine)
            report.evidence = [str(e).strip("[] ") for e in report.evidence]
            msg = AIMessage(content=report.model_dump_json())
            return {"messages": [msg], "report": report.model_dump(), "call_meta": meta}
        tools = MEMORY_TOOLS if s.get("n_tool_calls", 0) < cap else None
        out, meta = chain.invoke(s["messages"], tools=tools, config=cb)
        if not isinstance(out, AIMessage):
            out = AIMessage(content=_text(getattr(out, "content", out)))
        elif not out.tool_calls:      # keep the visible answer only (reasoning is still counted in the logs)
            out = AIMessage(content=_text(out.content), usage_metadata=out.usage_metadata,
                            response_metadata=out.response_metadata)
        if not tools and out.tool_calls:            # cap reached: ignore further tool requests
            out = AIMessage(content=_text(out.content) or "No further information.")
        return {"messages": [out], "call_meta": meta}

    def route_after_model(s: AgentState):
        return "tools" if tools_condition(s) == "tools" else "done"

    tool_node = ToolNode(MEMORY_TOOLS)

    def run_tools(s: AgentState, config: RunnableConfig):
        out = tool_node.invoke(s, config)
        return {"messages": out["messages"], "n_tool_calls": s.get("n_tool_calls", 0) + 1}

    def parse_citations(s: AgentState):
        delivered = s.get("delivered", [])
        if s.get("report"):
            answer = s["messages"][-1].content
            raw = s["report"].get("evidence", []) + CITE.findall(answer)
        else:
            answer = _text(s["messages"][-1].content)
            raw = CITE.findall(answer)
        cited = [m for m in dict.fromkeys(raw) if m in delivered]
        missing = not cited
        return {"answer": answer, "cited": cited or list(delivered), "citation_missing": missing}

    def write_answer(s: AgentState, config: RunnableConfig):
        rt = _rt(config)
        mem, agent = rt["memory"], s["agent"]
        kind = "observation" if task_kind(agent) in ("sensor", "history") else "derived"
        mid = mem.write(agent, s["answer"], derived_from=s["cited"], event_id=s["event"].get("id"), kind=kind)
        mem.after_turn(agent, s.get("delivered", []), [] if s.get("citation_missing") else s["cited"])
        return {"memory_id": mid}

    g = StateGraph(AgentState)
    g.add_node("read_context", read_context)
    g.add_node("call_model", call_model)
    g.add_node("tools", run_tools)
    g.add_node("parse_citations", parse_citations)
    g.add_node("write_answer", write_answer)
    g.add_edge(START, "read_context")
    g.add_edge("read_context", "call_model")
    g.add_conditional_edges("call_model", route_after_model, {"tools": "tools", "done": "parse_citations"})
    g.add_edge("tools", "call_model")
    g.add_edge("parse_citations", "write_answer")
    g.add_edge("write_answer", END)
    return g.compile()


# ======================= orchestrator =======================
class RunState(TypedDict, total=False):
    events: list[dict]
    i: int
    turn: int
    current: dict | None
    last: dict                                   # last Diagnosis output (for Report)
    answers: Annotated[list[dict], operator.add]
    turns: Annotated[list[dict], operator.add]   # per agent turn summary (for metrics/demo)
    maintenance_log: Annotated[list[dict], operator.add]


def build_orchestrator(checkpointer=None, interrupt_before_report: bool = False):
    agent_graph = build_agent_graph()

    def next_event(s: RunState, config: RunnableConfig):
        i = s.get("i", 0)
        if i >= len(s["events"]):
            return {"current": None}
        turn = _rt(config)["memory"].tick()
        return {"current": s["events"][i], "i": i + 1, "turn": turn}

    def route_event(s: RunState, config: RunnableConfig):
        ev = s.get("current")
        if ev is None:
            return "end"
        if _rt(config).get("mode", "replay") == "replay" and ev["type"] != "question":
            return "replay_write"
        return AGENT_FOR_EVENT.get(ev["type"], "sensor")

    def replay_write(s: RunState, config: RunnableConfig):
        ev = s["current"]
        mid = _rt(config)["memory"].write(ev["source_agent"], ev["text"], event_id=ev["id"], kind="observation")
        return {"turns": [{"turn": s["turn"], "agent": ev["source_agent"], "event_id": ev["id"],
                           "memory_id": mid, "replay": True}]}

    def make_runner(name: str):
        def run(s: RunState, config: RunnableConfig):
            rt = _rt(config)
            ev = s["current"]
            handoff = None
            agent = name
            if name == "sensor" and ev.get("source_agent") not in (None, "user", "sensor"):
                agent = ev["source_agent"]           # car plant: the zone's own agent
            if name == "report":
                if interrupt_before_report:
                    from langgraph.types import interrupt
                    interrupt({"question": ev["text"], "diagnosis": s["last"]})
                handoff = {"memory_id": s["last"]["memory_id"], "text": s["last"]["answer"],
                           "turn": s["last"]["turn"]}
            start = rt["logger"].mark()
            out = agent_graph.invoke({"agent": agent, "event": ev, "handoff": handoff}, config)
            rt["logger"].annotate_since(start, delivered=out.get("delivered", []), cited=out.get("cited", []),
                                        citation_missing=out.get("citation_missing", False),
                                        tool_rounds=out.get("n_tool_calls", 0))
            turn = {"turn": s["turn"], "agent": agent, "event_id": ev["id"], "memory_id": out["memory_id"],
                    "delivered": out.get("delivered", []), "cited": out.get("cited", []),
                    "context_tokens": out["context"]["tokens_used"], "answer": out["answer"],
                    "citation_missing": out.get("citation_missing", False),
                    "provider": (out.get("call_meta") or {}).get("provider"),
                    "fallback_used": (out.get("call_meta") or {}).get("fallback_used", False)}
            update: dict = {"turns": [turn]}
            if name == "diagnosis":
                update["last"] = {**turn}
            if name == "report":
                update["answers"] = [{
                    "event_id": ev["id"], "question": ev["text"], "gold": ev.get("gold"),
                    "hops": ev.get("hops"), "report": out.get("report"),
                    "report_memory_id": out["memory_id"], "diagnosis_memory_id": s["last"]["memory_id"],
                    "diagnosis": s["last"]["answer"], "turn": s["turn"]}]
            return update
        return run

    def maybe_maintain(s: RunState, config: RunnableConfig):
        n = _rt(config)["cfg"].maintenance.every_n_turns
        return "maintenance" if n and s["turn"] % n == 0 else "next"

    def maintenance(s: RunState, config: RunnableConfig):
        return {"maintenance_log": [_rt(config)["memory"].maintain()]}

    g = StateGraph(RunState)
    g.add_node("next_event", next_event)
    g.add_node("replay_write", replay_write)
    for name in ("sensor", "history", "diagnosis", "report"):
        g.add_node(name, make_runner(name))
    g.add_node("maintenance", maintenance)
    g.add_edge(START, "next_event")
    g.add_conditional_edges("next_event", route_event,
                            {"replay_write": "replay_write", "sensor": "sensor", "history": "history",
                             "diagnosis": "diagnosis", "end": END})
    g.add_edge("diagnosis", "report")
    for n in ("replay_write", "sensor", "history", "report"):
        g.add_conditional_edges(n, maybe_maintain, {"maintenance": "maintenance", "next": "next_event"})
    g.add_edge("maintenance", "next_event")
    return g.compile(checkpointer=checkpointer)


def sqlite_checkpointer(path: str | None):
    """LangGraph SqliteSaver: every orchestrator step is saved (inspection, time travel)."""
    if not path:
        return None
    from langgraph.checkpoint.sqlite import SqliteSaver
    p = resolve_path(path) if path != ":memory:" else path
    if p != ":memory:":
        p.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(p), check_same_thread=False))


def register_agents(memory, machines: list[str], zones: dict | None = None):
    core = ["history", "diagnosis", "report"] + ([] if zones else ["sensor"])
    for name in core:
        r = ROLES[name]
        memory.register_agent(name, r["role"], r["profile"] + (machines if name in ("sensor", "report") else []))
    for zid in zones or {}:                       # car plant: one monitoring agent per zone
        r = role_for(zid, zones)
        memory.register_agent(zid, r["role"], r["profile"])


def _prepare(scenario, memory, models, logger, cfg, mode, checkpointer, thread_id, token_budget):
    zones = scenario.get("zones") or {}
    register_agents(memory, scenario.get("machines", []), zones)
    app = build_orchestrator(checkpointer, cfg.get("demo", {}).get("interrupt_before_report", False))
    config = {"configurable": {"thread_id": thread_id or scenario["scenario_id"], "memory": memory,
                               "models": models, "logger": logger, "cfg": cfg, "mode": mode,
                               "token_budget": token_budget, "zones": zones},
              "recursion_limit": 6 * len(scenario["events"]) + 50}
    init = {"events": scenario["events"], "i": 0, "turn": 0, "answers": [], "turns": [], "maintenance_log": []}
    return app, config, init


def run_scenario(scenario: dict, memory, models: dict, logger, cfg, mode: str = "replay",
                 checkpointer=None, thread_id: str | None = None, token_budget: int | None = None) -> dict:
    """Runs one scenario through the orchestrator graph; returns the final RunState."""
    app, config, init = _prepare(scenario, memory, models, logger, cfg, mode, checkpointer, thread_id, token_budget)
    return app.invoke(init, config)


def stream_scenario(scenario: dict, memory, models: dict, logger, cfg, mode: str = "replay",
                    checkpointer=None, thread_id: str | None = None, token_budget: int | None = None):
    """Same as run_scenario but yields LangGraph `updates` events (for the live demo)."""
    app, config, init = _prepare(scenario, memory, models, logger, cfg, mode, checkpointer, thread_id, token_budget)
    yield from app.stream(init, config, stream_mode="updates")
