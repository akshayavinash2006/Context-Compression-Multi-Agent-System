"""Task-aware compression for agent handoffs and retrieved working context.

This is intentionally extractive and conservative: semantic units are ranked, then
the residual is compacted with the existing rule compressor. Originals remain in
GraphMemory's cold store and are addressable through their memory IDs.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from graphmem.compress import NEG, NUM, RuleCompressor

ENTITY = re.compile(r"\b[A-Z][A-Za-z0-9_.:/-]*\b|\b[A-Z]{2,}[A-Z0-9_-]*\b")
CONDITION = re.compile(r"\b(if|only if|unless|except|must|cannot|required|constraint|because|therefore|due to)\b", re.I)
CODE = re.compile(r"```[\s\S]*?```|`[^`]+`", re.M)
DEFAULT_WEIGHTS = {"relevance": .30, "density": .20, "novelty": .15, "dependency": .15, "criticality": .20}
NON_ENTITY_CAPITALS = {"a", "an", "the", "if", "only", "because", "unless", "when", "while", "it", "this"}


@dataclass
class ContextCompressor:
    count_tokens: object
    target_ratio: float = .30
    weights: dict | None = None
    max_iterations: int = 2

    def __post_init__(self):
        self.weights = dict(DEFAULT_WEIGHTS if self.weights is None else self.weights)
        if not 0 < self.target_ratio <= 1:
            raise ValueError("target_ratio must be in (0, 1]")

    @staticmethod
    def _units(text: str) -> list[str]:
        # Protect code blocks and split prose into sentence-sized units.
        blocks = []
        def hold(m):
            blocks.append(m.group(0)); return f"\u0000{len(blocks)-1}\u0000"
        protected = CODE.sub(hold, text)
        chunks = re.split(r"(?<=[.!?])\s+|\n+", protected)
        out = []
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk: continue
            chunk = re.sub(r"\u0000(\d+)\u0000", lambda m: blocks[int(m.group(1))], chunk)
            out.append(chunk)
        # Identical repeated statements contain no new fact and need not consume
        # budget multiple times. Keep the first occurrence to preserve ordering.
        return list(dict.fromkeys(out))

    def _score(self, unit: str, task_terms: set[str], prior: set[str]) -> dict:
        words = set(re.findall(r"[\w.-]+", unit.lower()))
        relevance = len(words & task_terms) / max(1, len(task_terms))
        density = min(1., (len(words) + 2 * len(NUM.findall(unit))) / max(1, len(unit.split())))
        facts = set(ENTITY.findall(unit)) | set(NUM.findall(unit))
        novelty = len(facts - prior) / max(1, len(facts)) if facts else .5
        negated = any(w in words or w.endswith("n't") for w in NEG)
        critical = min(1., .35 * bool(NUM.search(unit)) + .20 * bool(CONDITION.search(unit)) +
                       .25 * bool(CODE.search(unit)) + .20 * negated)
        dependency = min(1., .55 * bool(CONDITION.search(unit)) + .45 * bool(re.search(r"\b(because|therefore|so that|which means)\b", unit, re.I)))
        return {"relevance": relevance, "density": density, "novelty": novelty,
                "dependency": dependency, "criticality": critical}

    def __call__(self, text: str, task: str = "", metadata: dict | None = None) -> tuple[str, dict]:
        start = time.perf_counter()
        original = self.count_tokens(text)
        budget = max(1, int(original * self.target_ratio)) if original else 0
        if not text.strip() or original <= budget or self.target_ratio >= 1:
            return text, {"compressed": False, "original_tokens": original, "tokens": original,
                          "ratio": 1.0, "budget": budget, "warning": None, "retention": 1.0,
                          "selected": self._units(text), "residual": [], "missing": [],
                          "latency_s": time.perf_counter() - start}
        units = self._units(text)
        task_terms = set(re.findall(r"\w+", task.lower()))
        prior, scored = set(), []
        for i, unit in enumerate(units):
            dims = self._score(unit, task_terms, prior)
            score = sum(self.weights[k] * dims[k] for k in self.weights)
            facts = set(ENTITY.findall(unit)) | set(NUM.findall(unit))
            prior |= facts
            scored.append((i, unit, score, dims))
        # Keep dependencies and critical facts high in the ranking; stable ties preserve order.
        ranked = sorted(scored, key=lambda row: (row[2], row[3]["criticality"], -row[0]), reverse=True)
        chosen, selected_ids, used = [], set(), 0
        for i, unit, _, dims in ranked:
            cost = self.count_tokens(unit)
            if used + cost <= budget and (dims["criticality"] > 0 or not chosen or dims["relevance"] > 0):
                chosen.append((i, unit)); selected_ids.add(i); used += cost
        # Critical units are indivisible. Keep them even when that means reporting
        # an over-budget result; silently deleting a constraint or measurement is worse.
        for i, unit, _, dims in ranked:
            if i in selected_ids or dims["criticality"] <= 0:
                continue
            chosen.append((i, unit)); selected_ids.add(i); used += self.count_tokens(unit)
        selected = [u for _, u in sorted(chosen)]
        residual = [u for i, u in enumerate(units) if i not in selected_ids]
        # Preserve useful residual details by removing only filler/stop words, then
        # add as much as the remaining budget allows. The source text remains retrievable.
        rule = RuleCompressor({"not", "no", "never", "without", "before", "after", "if", "unless"})
        residual_lines = []
        for unit in residual:
            compact = rule.compress_prompt(unit, rate=.5)["compressed_prompt"].strip()
            if compact and compact != unit:
                residual_lines.append(compact)
        parts = list(selected)
        for unit in residual_lines:
            candidate = "\n".join(parts + [unit])
            if self.count_tokens(candidate) <= budget:
                parts.append(unit)
        result = "\n".join(parts)
        # Iteratively compact the selected/residual representation while guarding facts.
        for _ in range(max(0, self.max_iterations - 1)):
            if self.count_tokens(result) <= budget: break
            compact = rule.compress_prompt(result, rate=.5)["compressed_prompt"]
            if compact == result or not self._safe(result, compact): break
            result = compact
        missing = self._missing(text, result)
        tokens = self.count_tokens(result)
        warning = None
        if tokens > budget:
            warning = f"Token budget not met safely ({tokens} > {budget}); original remains available by reference."
        elif missing:
            warning = "Potential information loss detected; consult the original context by reference."
        return result, {"compressed": result != text, "original_tokens": original, "tokens": tokens,
                        "ratio": round(tokens / original, 4) if original else 1.0, "budget": budget,
                        "warning": warning, "retention": round(1 - len(missing) / max(1, len(self._critical_facts(text))), 4),
                        "selected": selected, "residual": residual_lines, "missing": missing,
                        "latency_s": round(time.perf_counter() - start, 6),
                        "original_reference": (metadata or {}).get("reference")}

    @staticmethod
    def _critical_facts(text: str) -> set[str]:
        words = re.findall(r"[a-z']+", text.lower())
        entities = {e for e in ENTITY.findall(text) if e.lower() not in NON_ENTITY_CAPITALS}
        return set(NUM.findall(text)) | entities | {w for w in words if w in NEG or w.endswith("n't")} | {m.group(0).lower() for m in CONDITION.finditer(text)}

    @classmethod
    def _missing(cls, original: str, compressed: str) -> list[str]:
        return sorted(f for f in cls._critical_facts(original) if f.lower() not in compressed.lower())

    @staticmethod
    def _safe(original: str, compressed: str) -> bool:
        if not set(NUM.findall(original)) <= set(NUM.findall(compressed)): return False
        words = re.findall(r"[a-z']+", original.lower())
        return all(w in compressed.lower() for w in words if w in NEG or w.endswith("n't"))
