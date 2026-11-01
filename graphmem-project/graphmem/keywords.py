"""W2: keyword extraction and normalisation (plan §8.2).

Union of four sources: regex domain IDs, domain vocabulary (spaCy PhraseMatcher or
a regex matcher), spaCy NER (heavy only), and KeyBERT / YAKE key phrases.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import yaml

from graphmem.config import resolve_path
from graphmem.runtime import heavy_available

PO_RE = re.compile(r"\bPO[\s\-]?(\d{3,6})\b", re.I)
GENERIC_ID_RE = re.compile(r"\b([A-Z]{2,5})[\s\-]?(\d{1,4})\b(?![.:]\d)")
STOP_PHRASE = {"hey", "team", "please", "note", "update", "report", "today", "yesterday",
               "shift", "the", "a", "an", "is", "are", "and", "of", "to", "why", "what",
               "should", "we", "do", "how", "which", "anything", "wrong"}


@dataclass
class Term:
    canonical: str
    kind: str          # machine | po | component | symptom | maintenance | logistics | entity | phrase
    surface: str


def simple_lemma(word: str) -> str:
    w = word.lower()
    if len(w) <= 3 or w.endswith(("ss", "us", "is", "ics")):
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ches", "shes", "xes")):
        return w[:-2]
    if w.endswith("s"):
        return w[:-1]
    return w


def canonical_id(prefix: str, digits: str) -> str:
    return f"{prefix.lower()}-{int(digits)}"


def digits(s: str) -> list[str]:
    return re.findall(r"\d+", s)


class KeywordExtractor:
    def __init__(self, cfg, embedder=None):
        self.cfg = cfg
        vocab_path = resolve_path(cfg.keywords.get("vocabulary", "domain/vocabulary.yaml"))
        with open(vocab_path, encoding="utf-8") as f:
            self.vocab = yaml.safe_load(f)
        prefixes = "|".join(sorted(self.vocab.get("machine_prefixes", []), key=len, reverse=True))
        self.machine_re = re.compile(rf"\b({prefixes})[\s\-]?(\d{{1,4}})\b(?![.:]\d)", re.I)
        self.term_kind: dict[str, str] = {}
        for kind in ("components", "symptoms", "maintenance", "logistics"):
            for t in self.vocab.get(kind, []):
                self.term_kind.setdefault(t.lower(), kind.rstrip("s") if kind != "maintenance" and kind != "logistics" else kind)
        self.aliases = {k.lower(): v.lower() for k, v in (self.vocab.get("aliases") or {}).items()}
        self.related = {k.lower(): [x.lower() for x in v] for k, v in (self.vocab.get("related") or {}).items()}
        surfaces = list(self.term_kind) + list(self.aliases)
        # longest first so "spindle bearing" is seen before "bearing"; optional plural suffix
        alt = "|".join(re.escape(s) for s in sorted(surfaces, key=len, reverse=True))
        self.vocab_re = re.compile(rf"\b({alt})(?:e?s)?\b", re.I)

        self.nlp = None
        if heavy_available("spacy"):
            try:
                import spacy
                self.nlp = spacy.load(cfg.keywords.spacy_model)
            except Exception:
                self.nlp = None
        self.kb = None
        if embedder is not None and getattr(embedder, "model", None) is not None and heavy_available("keybert"):
            from keybert import KeyBERT
            self.kb = KeyBERT(model=embedder.model)
        self.yake = None
        if cfg.keywords.use_yake_fallback:
            try:
                import yake
                self.yake = yake.KeywordExtractor(lan="en", n=2, top=10, dedupLim=0.7)
            except Exception:
                self.yake = None

    # ---------- normalisation ----------
    def normalize(self, phrase: str) -> str:
        p = re.sub(r"[^\w\s\-]", " ", phrase.lower())
        m = self.machine_re.fullmatch(p.strip())
        if m:
            return canonical_id(m.group(1), m.group(2))
        if self.nlp is not None:
            toks = [t.lemma_.lower() for t in self.nlp(p) if not t.is_space]
        else:
            toks = [simple_lemma(t) for t in p.split()]
        out = " ".join(toks).strip()
        return self.aliases.get(out, out)

    def kind_of(self, canonical: str) -> str:
        return self.term_kind.get(canonical, "phrase")

    # ---------- extraction ----------
    def extract(self, text: str, with_phrases: bool = True) -> list[Term]:
        terms: dict[str, Term] = {}

        def add(canon, kind, surface):
            if canon and canon not in terms:
                terms[canon] = Term(canon, kind, surface)

        # 1. regex IDs
        for m in PO_RE.finditer(text):
            add(f"po-{int(m.group(1))}", "po", m.group(0))
        for m in self.machine_re.finditer(text):
            add(canonical_id(m.group(1), m.group(2)), "machine", m.group(0))
        for m in GENERIC_ID_RE.finditer(text):
            if m.group(1).upper() != "PO":
                add(canonical_id(m.group(1), m.group(2)), "machine", m.group(0))
        # 2. vocabulary (regex matcher; spaCy lemmas are applied through normalize())
        for m in self.vocab_re.finditer(text):
            surf = m.group(1).lower()
            canon = self.aliases.get(surf, surf)
            add(canon, self.term_kind.get(canon, "phrase"), m.group(0))
        # 3. spaCy NER (heavy only)
        if self.nlp is not None:
            for ent in self.nlp(text).ents:
                if ent.label_ in self.cfg.keywords.ner_labels and not any(c.isdigit() for c in ent.text):
                    add(self.normalize(ent.text), "entity", ent.text)
        # 4. key phrases
        if with_phrases:
            for ph in self._phrases(text):
                canon = self.normalize(ph)
                words = canon.split()
                if not canon or any(c.isdigit() for c in canon) or all(w in STOP_PHRASE for w in words):
                    continue
                if canon in terms or all(w in terms or w in STOP_PHRASE for w in words):
                    continue
                add(canon, self.kind_of(canon), ph)
        out = list(terms.values())
        # domain terms first, then phrases; cap
        out.sort(key=lambda t: t.kind == "phrase")
        return out[: self.cfg.keywords.max_keywords_per_memory]

    def _phrases(self, text: str) -> list[str]:
        n = self.cfg.keywords.get("phrase_top_n", 3)
        if n <= 0:
            return []
        if self.kb is not None:
            kw = self.kb.extract_keywords(
                text, keyphrase_ngram_range=tuple(self.cfg.keywords.keybert_ngram),
                stop_words="english", top_n=self.cfg.keywords.keybert_top_n,
                use_mmr=self.cfg.keywords.keybert_use_mmr, diversity=self.cfg.keywords.keybert_diversity)
            phrases = [p for p, _ in kw]
            if phrases:
                return phrases[:n]
        if self.yake is not None:
            try:
                return [p for p, _ in self.yake.extract_keywords(text)][:n]
            except Exception:
                return []
        return []
