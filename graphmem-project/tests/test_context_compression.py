from graphmem.context_compression import ContextCompressor
from graphmem.tokens import TokenCounter


def compressor(ratio=.3):
    return ContextCompressor(TokenCounter(), target_ratio=ratio)


def test_numbers_dates_entities_and_negation_are_validated():
    text = ("CNC-3 vibration measured 7.2 mm/s on 2026-11-01. The model did not converge. "
            "The pump stopped because bearing temperature exceeded 80 C. " * 3)
    out, meta = compressor()(text, task="CNC-3 vibration diagnosis")
    assert "7.2" in out and "2026-11-01" in out and "not" in out
    assert meta["original_tokens"] > meta["tokens"]
    assert meta["ratio"] <= .30 or meta["warning"]


def test_condition_causality_and_technical_names_survive_when_budget_allows():
    text = ("Only if sensor X is calibrated, start the PyTorch job because TensorFlow failed. " * 2)
    out, meta = compressor(.8)(text, task="start the job")
    assert "Only if" in out and "because" in out
    assert "PyTorch" in out or meta["warning"]


def test_redundant_units_are_considered_and_small_context_is_unchanged():
    c = compressor(.3)
    small = "CNC-5 vibration normal."
    out, meta = c(small, task="vibration")
    assert out == small and meta["ratio"] == 1.0
    repeated = "CNC-5 vibration normal at 10:00. CNC-5 vibration normal at 10:00. CNC-5 temperature 40 C."
    out, meta = c(repeated, task="temperature")
    assert "40" in out or meta["warning"]


def test_invalid_ratio_rejected_and_overflow_exposed():
    try:
        compressor(0)
        assert False
    except ValueError:
        pass
    text = "```python\n" + "print('important')\n" * 30 + "```"
    out, meta = compressor(.05)(text, task="execute code")
    assert meta["original_tokens"] > 0
    assert meta["tokens"] == TokenCounter()(out)
    assert meta["warning"] or meta["tokens"] <= meta["budget"]


def test_validation_reports_missing_fact():
    assert "7.2" in ContextCompressor._missing("CNC-3 measured 7.2 mm/s", "CNC-3 measured 6 mm/s")
