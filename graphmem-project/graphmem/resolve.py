"""W2: resolve extracted terms to keyword nodes (exact -> fuzzy -> embedding -> new).

Digit guard: never match two keywords whose digit sequences differ (cnc-3 vs cnc-5).
"""
from __future__ import annotations

import re

import numpy as np
from rapidfuzz import fuzz, process

from graphmem.keywords import digits


def compact(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def kw_id(canonical: str) -> str:
    return f"K:{canonical}"


class KeywordIndex:
    def __init__(self, cfg, embedder, store):
        self.cfg, self.embed, self.store = cfg, embedder, store
        self.alias_map: dict[str, str] = {}       # surface/canonical -> keyword id
        self.compact_to_kw: dict[str, str] = {}
        self.ids: list[str] = []
        self._matrix: np.ndarray | None = None

    # ---------- index maintenance ----------
    def _invalidate(self):
        self._matrix = None

    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            texts = [self.store.g.nodes[k]["canonical"] for k in self.ids]
            self._matrix = self.embed.encode(texts) if texts else np.zeros((0, self.embed.dim), np.float32)
        return self._matrix

    def register(self, kid: str, canonical: str, aliases=()):
        if kid not in self.ids:
            self.ids.append(kid)
            self._invalidate()
        for a in {canonical, *aliases}:
            self.alias_map[a.lower()] = kid
            self.compact_to_kw[compact(a)] = kid

    def remove(self, kid: str):
        if kid in self.ids:
            self.ids.remove(kid)
            self._invalidate()
        for d in (self.alias_map, self.compact_to_kw):
            for k in [k for k, v in d.items() if v == kid]:
                del d[k]

    def redirect(self, old: str, new: str):
        """After a keyword merge: all aliases of `old` now point to `new`."""
        for d in (self.alias_map, self.compact_to_kw):
            for k, v in list(d.items()):
                if v == old:
                    d[k] = new
        if old in self.ids:
            self.ids.remove(old)
            self._invalidate()

    def lookup(self, canonical: str) -> str | None:
        return self.alias_map.get(canonical.lower())

    # ---------- resolution ----------
    def resolve(self, term: str, kind: str = "phrase", create: bool = True,
                surface: str | None = None) -> tuple[str | None, float]:
        """Returns (keyword id, match confidence). confidence 1.0 for exact/fuzzy/new."""
        term = term.lower()
        if term in self.alias_map:                                   # 1 exact
            kid = self.alias_map[term]
            if surface:
                self._add_alias(kid, surface)
            return kid, 1.0
        guard = self.cfg.resolution.digit_guard
        c = compact(term)
        if self.compact_to_kw and c:                                  # 2 fuzzy
            hit = process.extractOne(c, list(self.compact_to_kw), scorer=fuzz.ratio,
                                     score_cutoff=self.cfg.resolution.fuzzy_threshold)
            if hit:
                kid = self.compact_to_kw[hit[0]]
                if not guard or digits(term) == digits(self.store.g.nodes[kid]["canonical"]):
                    self._add_alias(kid, term)
                    return kid, 1.0
        if self.ids:                                                  # 3 embeddings
            sims = self.matrix() @ self.embed(term)
            j = int(np.argmax(sims))
            kid = self.ids[j]
            if sims[j] >= self.cfg.resolution.embed_threshold and (
                    not guard or digits(term) == digits(self.store.g.nodes[kid]["canonical"])):
                self._add_alias(kid, term)
                return kid, float(sims[j])
        if not create:
            return None, 0.0
        kid = kw_id(term)                                             # 4 new
        self.store.add_keyword(kid, term, kind, surface or term)
        self.register(kid, term, [surface] if surface else [])
        return kid, 1.0

    def nearest(self, term: str, top_k: int, min_cos: float) -> list[tuple[str, float]]:
        """Weaker seeds for query terms without a match (R1 step 3)."""
        if not self.ids:
            return []
        sims = self.matrix() @ self.embed(term)
        order = np.argsort(-sims)[:top_k]
        guard = self.cfg.resolution.digit_guard
        out = []
        for j in order:
            kid = self.ids[int(j)]
            if sims[j] < min_cos:
                break
            if guard and digits(term) and digits(term) != digits(self.store.g.nodes[kid]["canonical"]):
                continue
            out.append((kid, float(sims[j])))
        return out

    def _add_alias(self, kid: str, surface: str):
        s = surface.lower()
        self.store.g.nodes[kid]["aliases"].add(s)
        self.alias_map.setdefault(s, kid)
        self.compact_to_kw.setdefault(compact(s), kid)
