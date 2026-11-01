"""Rolling summary: LLM-written running summary + messages since the last refresh."""
from __future__ import annotations

from agents.prompts import SUMMARY_PROMPT
from baselines.base import MemoryBackend


class RollingSummary(MemoryBackend):
    name = "summary"

    def __init__(self, cfg, counter=None, summariser=None, logger=None, every: int | None = None):
        super().__init__(cfg, counter)
        self.summariser = summariser      # ModelChain (role "summariser"); its calls count as cost
        self.logger = logger
        self.every = every or cfg.experiment.rolling_summary_every
        self.summary = ""
        self.since: list[dict] = []

    def on_write(self, msg):
        self.since.append(msg)
        if len(self.since) >= self.every:
            self.refresh()

    def refresh(self):
        if not self.since:
            return
        messages = SUMMARY_PROMPT.format_messages(
            summary=self.summary or "(empty)", messages="\n".join(self.line(m) for m in self.since))
        if self.summariser is None:
            self.summary = (self.summary + "\n" + "\n".join(m["text"] for m in self.since)).strip()
        else:
            if self.logger is not None:
                self.logger.set_context(turn=self.clock.now, agent="summariser", event_id=None)
            out, _ = self.summariser.invoke(messages, config={"callbacks": [self.logger]} if self.logger else None)
            self.summary = str(out.content).strip()
        self.counters["summaries"] += 1
        self.since = []

    def prefix_text(self):
        return f"SUMMARY OF EARLIER MESSAGES:\n{self.summary}" if self.summary else ""

    def select(self, agent, query, token_budget):
        return list(self.since)
