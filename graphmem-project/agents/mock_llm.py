"""Deterministic rule-based chat model (LangChain BaseChatModel) for tests and offline runs.

It behaves like a careful small LLM restricted to the MEMORY block: it finds the machine
in the task, reads the facts about that machine, applies root-cause rules, cites the IDs
it used, can call the expand_memory tool, and supports structured output for Report.
Accuracy therefore depends only on which facts the memory backend delivered.
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

from graphmem.models import ACTIONS, ROOT_CAUSES
from graphmem.tokens import TokenCounter

MACHINE = re.compile(r"\b(cnc|vmc|hmc|lathe|press|mill|grinder|robot|edm)[\s\-]?(\d{1,4})\b(?![.:]\d)", re.I)
LINE = re.compile(r"^\[(M\d+)(?: original)? \| ([^|\]]+) \| t=(\d+)\]\s*(.*)$")

CAUSE_CUES = {
    "misalignment": re.compile(r"coupling|alignment|misalign"),
    "imbalance": re.compile(r"off-cent(re|er)|unbalanc|imbalance"),
    "coolant_failure": re.compile(r"coolant"),
    "overload": re.compile(r"heavy|deep cut|roughing|overload"),
    "lubrication_failure": re.compile(r"lubrication|oil level"),
    "sensor_fault": re.compile(r"calibration|jumped"),
    "belt_wear": re.compile(r"belt.*(crack|worn|overdue)|crack.*belt"),
}
SYMPTOMS = {
    "vibration": re.compile(r"vibrat\w*.*(above|exceed|rose|jump|over)|(above|exceed|over).*vibrat"),
    "temperature": re.compile(r"temperature.*(above|rising|over)|(above|over).*°c"),
    "current": re.compile(r"current.*(above|over)"),
    "speed": re.compile(r"fluctuat"),
}
QUESTION_SYMPTOMS = {"vibration": r"vibrat|shak", "temperature": r"hot|temperature|overheat",
                     "current": r"current|amps", "speed": r"speed|fluctuat|unstable"}
COMPATIBLE = {
    "vibration": {"bearing_wear", "misalignment", "imbalance", "sensor_fault"},
    "temperature": {"coolant_failure", "lubrication_failure"},
    "current": {"overload"},
    "speed": {"belt_wear"},
}
NORMAL = re.compile(r"\b(normal|within|no issues?|ok|healthy)\b")
ABNORMAL = re.compile(r"\b(above|exceed\w*|dropped|low|crack\w*|overdue|rising)\b")
COMPONENTS = ["spindle bearing", "bearing", "coolant pump", "coolant", "coupling", "fixture",
              "lubrication", "belt", "sensor", "motor"]


def canon_machine(m: re.Match) -> str:
    return f"{m.group(1).lower()}-{int(m.group(2))}"


def parse_lines(text: str) -> list[dict]:
    out = []
    for raw in text.splitlines():
        m = LINE.match(raw.strip())
        if m:
            out.append({"id": m.group(1), "agent": m.group(2).strip(), "turn": int(m.group(3)),
                        "text": m.group(4), "low": m.group(4).lower()})
    return out


def bearing_overdue(low: str) -> bool:
    if "bearing" not in low:
        return False
    if "overdue" in low:
        return True
    a = re.search(r"replaced (\d+) months", low)
    b = re.search(r"interval (?:is |of )?(\d+) months", low)
    return bool(a and b and int(a.group(1)) > int(b.group(1)))


class Diagnoser:
    """Root-cause rules over the facts about one machine."""

    def __init__(self, lines: list[dict], task: str):
        self.task = task
        found = [canon_machine(m) for m in MACHINE.finditer(task)]
        if not found:
            counts: dict[str, int] = {}
            for ln in lines:
                for m in MACHINE.finditer(ln["text"]):
                    counts[canon_machine(m)] = counts.get(canon_machine(m), 0) + 1
            found = sorted(counts, key=lambda k: -counts[k])[:1]
        self.machine = found[0] if found else ""
        rel = [ln for ln in lines if self.machine and self.machine in
               {canon_machine(m) for m in MACHINE.finditer(ln["text"])}]
        self.lines = self._apply_corrections(rel)

    @staticmethod
    def _apply_corrections(lines):
        drop = set()
        for ln in lines:
            if "correction" in ln["low"]:
                comps = [c for c in COMPONENTS if c in ln["low"]]
                for other in lines:
                    if other is not ln and other["turn"] < ln["turn"] and any(c in other["low"] for c in comps[:1]):
                        drop.add(other["id"])
        return [ln for ln in lines if ln["id"] not in drop]

    def diagnose(self) -> tuple[str, list[str]]:
        symptoms: dict[str, list[str]] = {}
        for name, pat in QUESTION_SYMPTOMS.items():
            if re.search(pat, self.task.lower()):
                symptoms.setdefault(name, [])
        scores = {c: 0.0 for c in ROOT_CAUSES if c != "none"}
        evidence = {c: [] for c in scores}
        for ln in self.lines:
            low = ln["low"]
            for name, pat in SYMPTOMS.items():
                if pat.search(low):
                    symptoms.setdefault(name, []).append(ln["id"])
            if NORMAL.search(low) and not ABNORMAL.search(low):
                continue
            if bearing_overdue(low):
                scores["bearing_wear"] += 2
                evidence["bearing_wear"].append(ln["id"])
            for cause, pat in CAUSE_CUES.items():
                if pat.search(low):
                    scores[cause] += 2
                    evidence[cause].append(ln["id"])
            # an explicit label (e.g. from the Diagnosis hand-off) counts as a cue
            for cause in scores:
                if cause in low:
                    scores[cause] += 3
                    evidence[cause].append(ln["id"])
        for sym, ids in symptoms.items():
            for cause in COMPATIBLE[sym]:
                if scores[cause] > 0:
                    scores[cause] += 1
                    evidence[cause] += ids
        best = max(scores, key=lambda c: (scores[c], -ROOT_CAUSES.index(c)))
        if scores[best] < 2:
            return "none", []
        return best, list(dict.fromkeys(evidence[best]))


class RuleBasedChatModel(BaseChatModel):
    """Offline stand-in for an LLM. expand_policy: never | short | always."""

    expand_policy: str = "short"
    model_name: str = "rule-based"

    @property
    def _llm_type(self) -> str:
        return "rule-based-mock"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "expand_policy": self.expand_policy}

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        formatted = [convert_to_openai_tool(t) for t in tools]
        return self.bind(tools=formatted, tool_choice=tool_choice, **kwargs)

    # ---------- generation ----------
    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        tools = [t["function"]["name"] for t in kwargs.get("tools") or []]
        human = next((m for m in messages if isinstance(m, HumanMessage)), None)
        text = str(human.content) if human else ""
        context, task = self._split(text)
        lines = parse_lines(context)
        for m in messages:                       # expanded originals arrive as ToolMessages
            if isinstance(m, ToolMessage):
                lines += parse_lines(str(m.content))
        already_expanded = any(isinstance(m, ToolMessage) for m in messages)

        if "ReportSchema" in tools:
            msg = self._report(lines, task, structured=True)
        elif text.startswith("SUMMARISE"):
            msg = AIMessage(content=self._summarise(text))
        elif "expand_memory" in tools and not already_expanded and (target := self._want_expand(lines, task)):
            msg = AIMessage(content="", tool_calls=[{
                "name": "expand_memory", "args": {"memory_id": target}, "id": f"call_{target}"}])
        elif "Write the final report as JSON" in task:
            msg = self._report(lines, task, structured=False)
        elif task.startswith(("New input", "New maintenance")):
            body = task.split("\n", 2)[1] if "\n" in task else task
            msg = AIMessage(content=body.strip())
        else:
            msg = AIMessage(content=self._diagnose_text(lines, task))

        counter = TokenCounter()
        n_in = counter.count_messages(messages)
        n_out = counter(str(msg.content)) + sum(counter(json.dumps(tc["args"])) for tc in msg.tool_calls)
        msg.usage_metadata = {"input_tokens": n_in, "output_tokens": n_out, "total_tokens": n_in + n_out}
        msg.response_metadata = {"model_name": self.model_name, "provider": "mock"}
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @staticmethod
    def _split(text: str) -> tuple[str, str]:
        if "MEMORY:" in text and "TASK:" in text:
            ctx = text.split("MEMORY:", 1)[1].split("TASK:", 1)[0]
            return ctx.strip(), text.split("TASK:", 1)[1].strip()
        return "", text

    def _want_expand(self, lines, task) -> str | None:
        if self.expand_policy == "never" or not lines:
            return None
        d = Diagnoser(lines, task)
        rel = d.lines or lines
        if self.expand_policy == "always":
            return rel[0]["id"]
        for ln in rel:                       # "short": a fact that looks cut off
            if len(ln["text"].split()) < 5:
                return ln["id"]
        return None

    def _diagnose_text(self, lines, task) -> str:
        d = Diagnoser(lines, task)
        cause, ev = d.diagnose()
        cites = " ".join(f"[{e}]" for e in ev)
        if cause == "none":
            return f"No clear root cause found for {d.machine or 'the machine'} in the available facts. " \
                   f"Action: {ACTIONS['none']}."
        return (f"Likely root cause for {d.machine}: {cause} based on {cites}. "
                f"Action: {ACTIONS[cause]}.")

    def _report(self, lines, task, structured: bool) -> AIMessage:
        d = Diagnoser(lines, task)
        cause, ev = d.diagnose()
        args = {"machine": d.machine, "root_cause": cause, "action": ACTIONS[cause], "evidence": ev}
        if structured:
            return AIMessage(content="", tool_calls=[{"name": "ReportSchema", "args": args, "id": "call_report"}])
        return AIMessage(content=json.dumps(args))

    @staticmethod
    def _summarise(text: str) -> str:
        cur = text.split("CURRENT SUMMARY:", 1)[1].split("NEW MESSAGES:", 1)[0].strip()
        new = text.split("NEW MESSAGES:", 1)[1].split("Write the updated summary", 1)[0]
        keep = [ln for ln in cur.splitlines() if ln.strip() and ln.strip() != "(empty)"]
        for ln in new.splitlines():
            m = LINE.match(ln.strip())
            if m:
                short = " ".join(m.group(4).split()[:15])
                keep.append(f"[{m.group(1)} | {m.group(2).strip()} | t={m.group(3)}] {short}")
        keep = list(dict.fromkeys(keep))[-30:]
        return "\n".join(keep)
