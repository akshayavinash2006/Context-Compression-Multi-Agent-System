from types import SimpleNamespace

import pytest

from eval import benchmark_qwen3


def test_qwen3_benchmark_requires_exact_local_model(monkeypatch):
    expected = {role: SimpleNamespace(primary="ollama:qwen3:8b") for role in
                ("sensor", "history", "diagnosis", "report", "summariser")}
    monkeypatch.setattr(benchmark_qwen3, "get_models", lambda *_args, **_kwargs: expected)
    benchmark_qwen3.verify_local_model(object())


def test_qwen3_benchmark_refuses_fallback(monkeypatch):
    chains = {role: SimpleNamespace(primary="ollama:qwen2.5:7b") for role in
              ("sensor", "history", "diagnosis", "report", "summariser")}
    monkeypatch.setattr(benchmark_qwen3, "get_models", lambda *_args, **_kwargs: chains)
    with pytest.raises(SystemExit, match="will not mix models"):
        benchmark_qwen3.verify_local_model(object())
