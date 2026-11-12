# Test and Qwen 3 Benchmark Report

**Date:** 2026-11-12  
**Project:** GraphMem multi-agent context compression system

## Test run

Command:

```powershell
python -m pytest -q
```

Result: **64 passed, 1 failed** in 24.87 seconds.

The failing test is `tests/test_trace.py::test_trace_through_pipeline`. The worked-example answer is still `bearing_wear`, but the produced report trace does not contain a memory written by `sensor`; the test expects the trace to include `report`, `diagnosis`, `sensor`, and `history`. This is an outstanding trace coverage failure. It is reported here as observed; the test was not weakened or changed to make the suite green.

The targeted compression, agent graph, and Qwen benchmark tests also passed: **15 passed** with:

```powershell
python -m pytest -q tests/test_qwen3_benchmark.py tests/test_context_compression.py tests/test_compress.py tests/test_graphs.py
```

## Local Qwen 3 benchmark

The benchmark compared `full` history against `graph` memory on the simulator's one-question `worked_example`, seed 42, replay mode. Every recorded model call used `qwen3:8b`; **0 fallback calls** were recorded.

| Condition | Accuracy | Input tokens per task | Latency per question | Action score | Trace precision | Trace recall |
|---|---:|---:|---:|---:|---:|---:|
| Full history | 1.00 | 818 | 48.91 s | 0.67 | 0.67 | 1.00 |
| GraphMem | 1.00 | 707 | 41.21 s | 1.00 | 0.67 | 1.00 |

GraphMem used **111 fewer input tokens (13.6%)** in this one-question smoke run. The sample is too small to support a general accuracy or speed claim. Run details, calls, answers, traces, and derived metrics are in `results/qwen3_8b_smoke/run/`.

Reproduce the small run:

```powershell
python -m eval.benchmark_qwen3 --scenarios results/qwen3_8b_smoke/worked_example.json --seeds 42 --out results/qwen3_8b_smoke/run
```

Run a wider dev benchmark with three seeds:

```powershell
python -m eval.benchmark_qwen3 --scenarios data/scenarios/dev --seeds 42 43 44 --out results/qwen3_8b
```

## API and runtime requirements

No hosted API key is needed. The benchmark connects to the local Ollama service and uses the installed `qwen3:8b` model. Ollama must be running on `127.0.0.1:11434`.
