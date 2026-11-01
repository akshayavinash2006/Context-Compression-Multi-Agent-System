"""Token counting with ONE fixed tokenizer for every condition and provider (plan §13.6).

tiktoken (cl100k_base) when available; otherwise a regex estimate that is close
to BPE counts for English text.
"""
from __future__ import annotations

import functools
import re

_FALLBACK = re.compile(r"\w+|[^\w\s]")


@functools.lru_cache(maxsize=4)
def _encoding(name: str):
    try:
        import tiktoken
        return tiktoken.get_encoding(name)
    except Exception:          # not installed, or BPE file cannot be downloaded
        return None


class TokenCounter:
    def __init__(self, spec: str = "tiktoken:cl100k_base"):
        kind, _, name = spec.partition(":")
        self.enc = _encoding(name or "cl100k_base") if kind == "tiktoken" else None
        self.name = spec if self.enc else "regex-estimate"

    def __call__(self, text: str) -> int:
        if not text:
            return 0
        if self.enc is not None:
            return len(self.enc.encode(text, disallowed_special=()))
        return len(_FALLBACK.findall(text))

    def count_messages(self, messages) -> int:
        """Tokens of a chat prompt: content of every message + 4 per message overhead."""
        total = 0
        for m in messages:
            content = getattr(m, "content", m)
            if isinstance(content, list):  # multi-part content (e.g. Gemini)
                content = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
            total += self(str(content)) + 4
            for tc in getattr(m, "tool_calls", None) or []:
                total += self(str(tc.get("name", ""))) + self(str(tc.get("args", "")))
        return total


_default: TokenCounter | None = None


def count_tokens(text: str) -> int:
    global _default
    if _default is None:
        _default = TokenCounter()
    return _default(text)
