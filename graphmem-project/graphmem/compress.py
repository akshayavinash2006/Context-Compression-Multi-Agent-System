"""W1: compress agent output (plan §8.1).

Heavy: LLMLingua-2 token classification. Light fallback: rule-based removal of
filler words and stop words. Both keep numbers and negations, and both go through
the same safety check.
"""
from __future__ import annotations

import re

from graphmem.runtime import heavy_available

NEG = {"not", "no", "never", "none", "without", "n't", "nothing", "nor"}
NUM = re.compile(r"\d+(?:\.\d+)?")

FILLER_PHRASES = [
    r"\bhey (?:team|guys|all|folks)\b[,!]?", r"\bhi (?:team|all)\b[,!]?", r"\bfyi\b[,:]?",
    r"\bplease note(?: that)?\b", r"\bjust (?:a )?(?:quick )?(?:heads[- ]up|update)\b[,:]?",
    r"\bi think\b", r"\bit seems(?: that)?\b", r"\bas far as i can tell\b", r"\bbasically\b",
    r"\bactually\b", r"\bkind of\b", r"\bsort of\b", r"\bto be honest\b", r"\bjust\b",
]
STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "that", "this", "which", "really", "very", "so", "also", "please", "kindly", "we", "our",
    "i", "you", "it", "its", "of", "to", "and", "there", "here", "currently", "today", "now,",
    "right", "quite", "some", "any", "do", "does", "did", "will", "would", "should", "could",
    "can", "may", "might", "then", "than", "for", "on", "in", "at", "by", "with", "from",
}


class RuleCompressor:
    """Drops filler phrases and stop words; never drops numbers, negations or forced tokens."""

    def __init__(self, force_tokens):
        self.keep = {t.lower() for t in force_tokens} | NEG

    def compress_prompt(self, text: str, rate: float = 0.5, **_) -> dict:
        if rate >= 1.0:
            return {"compressed_prompt": text}
        out = text
        for pat in FILLER_PHRASES:
            out = re.sub(pat, " ", out, flags=re.I)
        kept = []
        for tok in out.split():
            core = re.sub(r"^[^\w%]+|[^\w%]+$", "", tok).lower()
            if core in STOP and core not in self.keep and not any(ch.isdigit() for ch in core):
                # keep trailing sentence punctuation
                if tok and tok[-1] in ".;:," and kept:
                    kept[-1] = kept[-1].rstrip(".;:,") + tok[-1]
                continue
            kept.append(tok)
        comp = " ".join(kept).strip(" ,")
        return {"compressed_prompt": comp or text}


class Compressor:
    def __init__(self, cfg, count_tokens, backend=None):
        self.cfg, self.count = cfg, count_tokens
        self.backend_name = "rule"
        if backend is not None:
            self.pc = backend
            self.backend_name = "custom"
        elif heavy_available("llmlingua"):
            from llmlingua import PromptCompressor
            model = cfg.model_small if cfg.use_small_model else cfg.model
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
            self.pc = PromptCompressor(model_name=model, use_llmlingua2=True, device_map=device)
            self.backend_name = "llmlingua2"
        else:
            self.pc = RuleCompressor(cfg.force_tokens)

    @staticmethod
    def safe(orig: str, comp: str) -> bool:
        nums_ok = set(NUM.findall(orig)) <= set(NUM.findall(comp))
        orig_words = re.findall(r"[a-z']+", orig.lower())
        negs = {w for w in orig_words if w in NEG or w.endswith("n't")}
        comp_l = comp.lower()
        negs_ok = all(n in comp_l for n in negs)
        return nums_ok and negs_ok

    def __call__(self, text: str) -> tuple[str, dict]:
        n = self.count(text)
        if n < self.cfg.min_tokens_to_compress or self.cfg.rate >= 1.0:
            return text, {"compressed": False, "orig_tokens": n, "tokens": n}
        rate = self.cfg.rate
        while rate < 1.0 + 1e-9:
            out = self.pc.compress_prompt(
                text, rate=rate,
                force_tokens=list(self.cfg.force_tokens),
                force_reserve_digit=self.cfg.force_reserve_digit,
                drop_consecutive=True)
            comp = out["compressed_prompt"].strip()
            if comp and (not self.cfg.safety_check or self.safe(text, comp)):
                return comp, {"compressed": True, "rate": round(rate, 2),
                              "orig_tokens": n, "tokens": self.count(comp)}
            rate += self.cfg.fallback_rate_step
        return text, {"compressed": False, "fallback": True, "orig_tokens": n, "tokens": n}
