"""Shared embedding model with a cache (plan §3).

Heavy: sentence-transformers all-MiniLM-L6-v2.
Light fallback: deterministic hashed word + character-trigram vectors.
"""
from __future__ import annotations

import re
import zlib

import numpy as np

from graphmem.runtime import heavy_available

_WORD = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


class HashEmbedder:
    """Bag of hashed word unigrams, word bigrams and character trigrams, L2-normalised."""

    kind = "hash"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _features(self, text: str):
        words = _WORD.findall(text.lower())
        for w in words:
            yield "w:" + w, 1.0
            stem = w[:-1] if len(w) > 4 and w.endswith("s") else w
            padded = f"#{stem}#"
            for i in range(len(padded) - 2):
                yield "c:" + padded[i:i + 3], 0.3
        for a, b in zip(words, words[1:]):
            yield f"b:{a}_{b}", 0.5

    def encode_one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for feat, weight in self._features(text):
            h = zlib.crc32(feat.encode())
            v[h % self.dim] += weight if (h >> 16) & 1 else -weight
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


class Embedder:
    def __init__(self, cfg, use_heavy: bool | None = None):
        self.cache: dict[str, np.ndarray] = {}
        self.model = None
        heavy = heavy_available("sentence_transformers") if use_heavy is None else use_heavy
        if heavy:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(cfg.embeddings.model)
            except Exception:
                self.model = None
        self.kind = "minilm" if self.model is not None else "hash"
        self._hash = HashEmbedder(cfg.embeddings.get("hash_dim", 512))
        self.dim = self.model.get_sentence_embedding_dimension() if self.model else self._hash.dim

    def encode(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if missing:
            if self.model is not None:
                vecs = self.model.encode(missing, normalize_embeddings=True, show_progress_bar=False)
            else:
                vecs = [self._hash.encode_one(t) for t in missing]
            for t, v in zip(missing, vecs):
                self.cache[t] = np.asarray(v, dtype=np.float32)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self.cache[t] for t in texts])

    def __call__(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def cos(self, a: str, b: str) -> float:
        return float(self(a) @ self(b))


def as_langchain_embeddings(embedder: Embedder):
    """Wrap the shared embedder as a LangChain `Embeddings` (used by the Vector RAG baseline)."""
    from langchain_core.embeddings import Embeddings

    class _LCEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return embedder.encode(list(texts)).tolist()

        def embed_query(self, text):
            return embedder(text).tolist()

    return _LCEmbeddings()
