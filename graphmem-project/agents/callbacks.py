"""CallLogger: LangChain callback handler that writes one record per LLM call (plan §13.6)."""
from __future__ import annotations

import json
import time
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler

from graphmem.tokens import TokenCounter


class CallLogger(BaseCallbackHandler):
    def __init__(self, counter: TokenCounter | None = None, **static):
        self.counter = counter or TokenCounter()
        self.static = static            # condition, scenario, seed
        self.context: dict = {}         # turn, agent (set by the agent runner)
        self.records: list[dict] = []
        self._pending: dict = {}
        self.listener = None          # optional callable(kind, payload): live UI events

    def _emit(self, rec: dict):
        if self.listener is not None:
            try:
                self.listener("llm", rec)
            except Exception:
                pass

    def on_chat_model_start_notify(self, rec: dict):
        if self.listener is not None:
            try:
                self.listener("llm_start", rec)
            except Exception:
                pass

    def set_context(self, **ctx):
        self.context = ctx

    # ---------- LangChain hooks ----------
    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        msgs = messages[0] if messages else []
        self._pending[run_id] = {
            "t0": time.perf_counter(),
            "input_tokens": self.counter.count_messages(msgs),
            "provider": (metadata or {}).get("provider") or (metadata or {}).get("ls_provider"),
            "model": (metadata or {}).get("model") or (metadata or {}).get("ls_model_name"),
            "fallback_index": (metadata or {}).get("fallback_index", 0),
        }
        p = self._pending[run_id]
        self.on_chat_model_start_notify({**self.static, **self.context, "provider": p["provider"],
                                         "model": p["model"], "input_tokens": p["input_tokens"]})

    def on_llm_end(self, response, *, run_id, **kwargs):
        p = self._pending.pop(run_id, None)
        if p is None:
            return
        usage, n_tool_calls, text = {}, 0, ""
        try:
            gen = response.generations[0][0]
            msg = getattr(gen, "message", None)
            usage = dict(getattr(msg, "usage_metadata", None) or {})
            n_tool_calls = len(getattr(msg, "tool_calls", None) or [])
            text = str(getattr(msg, "content", "") or gen.text)
        except (IndexError, AttributeError):
            pass
        self.records.append({
            **self.static, **self.context,
            "provider": p["provider"], "model": p["model"],
            "fallback_used": bool(p["fallback_index"]),
            "input_tokens": p["input_tokens"],
            "provider_input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens", self.counter(text)),
            "latency_s": round(time.perf_counter() - p["t0"], 4),
            "tool_calls": n_tool_calls,
            "error": None,
        })
        self._emit(self.records[-1])

    def on_llm_error(self, error, *, run_id, **kwargs):
        p = self._pending.pop(run_id, None)
        if p is None:
            return
        self.records.append({**self.static, **self.context, "provider": p["provider"], "model": p["model"],
                             "fallback_used": bool(p["fallback_index"]), "input_tokens": p["input_tokens"],
                             "provider_input_tokens": None, "output_tokens": 0,
                             "latency_s": round(time.perf_counter() - p["t0"], 4), "tool_calls": 0,
                             "error": f"{type(error).__name__}: {str(error)[:200]}"})
        self._emit(self.records[-1])

    # ---------- helpers ----------
    def mark(self) -> int:
        return len(self.records)

    def annotate_since(self, start: int, **fields):
        for r in self.records[start:]:
            r.update(fields)

    def dump(self, path: str | Path):
        with open(path, "a", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r, default=str) + "\n")
