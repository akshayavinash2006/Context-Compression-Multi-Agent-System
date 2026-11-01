import re

from graphmem.compress import Compressor
from graphmem.tokens import TokenCounter

LONG = ("Hey team, just a quick heads-up: the CNC-3 spindle vibration was really at 7.2 mm/s at 14:05, "
        "which is above the 6 mm/s limit, and the coolant pump is not running. I think we should basically "
        "check it during the shift. Let me know if you need more details.")


def test_numbers_and_negations_survive(cfg):
    c = Compressor(cfg.compression, TokenCounter())
    comp, meta = c(LONG)
    assert meta["compressed"] is True
    assert meta["tokens"] < meta["orig_tokens"]
    for n in ["3", "7.2", "14:05", "6"]:
        assert n in comp
    assert "not" in comp.lower()
    assert Compressor.safe(LONG, comp)


def test_short_text_untouched(cfg):
    c = Compressor(cfg.compression, TokenCounter())
    text = "CNC-5 vibration normal."
    comp, meta = c(text)
    assert comp == text and meta["compressed"] is False


class DropDigits:
    """A compressor that loses numbers: the safety check must reject it."""

    def compress_prompt(self, text, rate=0.5, **_):
        return {"compressed_prompt": re.sub(r"\d", "", text)}


def test_fallback_when_number_dropped(cfg):
    c = Compressor(cfg.compression, TokenCounter(), backend=DropDigits())
    comp, meta = c(LONG)
    assert comp == LONG
    assert meta.get("fallback") is True and meta["compressed"] is False


def test_rate_one_disables_compression(cfg):
    sub = cfg.copy()
    sub.compression.rate = 1.0
    comp, meta = Compressor(sub.compression, TokenCounter())(LONG)
    assert comp == LONG and not meta["compressed"]
