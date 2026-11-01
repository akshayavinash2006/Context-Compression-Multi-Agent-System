"""Contradiction probability for W3 (plan §8.3).

Heavy: NLI cross-encoder (output order [contradiction, entailment, neutral]).
Light fallback: heuristic over numbers, negations, antonyms and correction cues.
"""
from __future__ import annotations

import re

import numpy as np

from graphmem.runtime import heavy_available

NUM = re.compile(r"\d+(?:\.\d+)?")
# machine IDs (CNC-3, CNC 3, cnc-3) and clock times are not measured values
NOT_A_VALUE = re.compile(r"\b[A-Z]{2,6}[\s\-]?\d+\b(?![.:]\d)|\b[a-z]{2,6}-\d+\b|\b\d{1,2}:\d{2}\b")
NEG = {"not", "no", "never", "none", "without"}
CORRECTION = re.compile(r"\b(correction|corrected|actually|revised|update[ds]?|instead|now)\b", re.I)
WORD = re.compile(r"[a-z]+")
MEM_ID = re.compile(r"\[?\bM\d+\b\]?")


def _softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class HeuristicNLI:
    kind = "heuristic"

    def __init__(self, antonyms=()):
        self.antonyms = [tuple(p) for p in antonyms]

    def contradiction(self, a: str, b: str) -> float:
        a, b = MEM_ID.sub(" ", a), MEM_ID.sub(" ", b)         # memory citations are not facts
        wa, wb = set(WORD.findall(a.lower())), set(WORD.findall(b.lower()))
        overlap = len(wa & wb) / max(1, min(len(wa), len(wb)))
        if overlap < 0.3:
            return 0.05
        score = 0.05
        na, nb = set(NUM.findall(NOT_A_VALUE.sub(" ", a))), set(NUM.findall(NOT_A_VALUE.sub(" ", b)))
        if na and nb and na != nb:
            score = max(score, 0.8)
        if bool(wa & NEG) != bool(wb & NEG):
            score = max(score, 0.75)
        for x, y in self.antonyms:
            if (x in wa and y in wb and y not in wa) or (y in wa and x in wb and x not in wa):
                score = max(score, 0.75)
        if score > 0.5 and (CORRECTION.search(a) or CORRECTION.search(b)):
            score = max(score, 0.9)
        return score


class CrossEncoderNLI:
    kind = "cross-encoder"

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def contradiction(self, a: str, b: str) -> float:
        logits = np.asarray(self.model.predict([(a, b), (b, a)]))     # (2, 3)
        probs = _softmax(logits)
        return float(probs[:, 0].max())                                # index 0 = contradiction


def make_nli(cfg, antonyms=()):
    if heavy_available("sentence_transformers"):
        try:
            name = cfg.merge.nli_model_small if cfg.merge.get("use_small_nli") else cfg.merge.nli_model
            return CrossEncoderNLI(name)
        except Exception:
            pass
    return HeuristicNLI(antonyms)
