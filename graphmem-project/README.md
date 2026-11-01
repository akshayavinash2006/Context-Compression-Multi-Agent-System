# GraphMem: shared graph memory for multi-agent LLM systems

Implementation of `../graph_memory_implementation_plan.md`. Four agents (Sensor, History,
Diagnosis, Report) diagnose factory faults. Instead of re-reading one flat chat history,
they share a **memory graph**: every message is compressed, keyworded, deduplicated and
linked, and each agent gets a small ranked context (Personalized PageRank + token budget).
Every answer can be traced back to its source facts.

Everything is built on **LangGraph** and **LangChain**:

| Piece | LangGraph / LangChain |
|---|---|
| Write path W1–W4 | `StateGraph` with a merge / update / create branch (`graphmem/graphs/write_graph.py`) |
| Read path R1–R4 | `StateGraph` with PPR or vector-fallback branch (`graphmem/graphs/read_graph.py`) |
| Maintenance | `StateGraph` archive → merge keywords → cleanup (`graphmem/graphs/maintain_graph.py`) |
| Agents | Subgraph `read_context → call_model ⇄ ToolNode → parse_citations → write_answer` (`agents/graph.py`) |
| Orchestrator | `StateGraph` over scenario events, `SqliteSaver` checkpoints, optional `interrupt()` before Report |
| Models | `init_chat_model` for Ollama, Groq, Gemini, OpenRouter; rate limiters; fallback chain; `SQLiteCache` |
| Tools | `expand_memory`, `trace_memory` (`@tool`, memory injected from the run config) |
| Report | `with_structured_output(ReportSchema)` |
| Retrieval | every backend exposed as a `BaseRetriever`; Vector RAG on `InMemoryVectorStore` |
| Logging | `CallLogger(BaseCallbackHandler)` → `calls.jsonl` |

## Setup (Windows)

```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt            # enough for everything in mock mode
pip install -r requirements-models.txt     # optional: LLMLingua-2, spaCy, KeyBERT, MiniLM, NLI
python -m spacy download en_core_web_sm    # if you installed the models
copy .env.example .env                     # API keys for the multi_provider preset
```

LLMs: install [Ollama](https://ollama.com), then `ollama pull qwen2.5:7b` and `ollama pull qwen2.5:3b`.
For free-tier APIs, put `GROQ_API_KEY` / `GOOGLE_API_KEY` / `OPENROUTER_API_KEY` in `.env`.

## Presets (`--preset`)

| Preset | Agents run on |
|---|---|
| `mock` | Deterministic rule-based chat model; no GPU, no network (tests, quick dev runs) |
| `single_provider` | All agents on local Ollama `qwen2.5:7b` (main controlled experiments) |
| `multi_provider` | Sensor: Ollama 3B, History: Ollama 7B, Diagnosis: Groq Llama-3.3-70B, Report: Gemini Flash |

Unavailable models (no key, no server) are skipped with a printed reason and the next
model in the fallback chain is used (Ollama, then mock).

Heavy NLP models are used automatically when installed (`runtime.use_heavy_models: auto`);
otherwise light fallbacks are used: rule compressor, hashed embeddings, heuristic NLI, YAKE.

## Commands

```powershell
pytest -q                                                    # 40 tests, mock + light fallbacks
python -m data.simulator --n 10 --split dev                  # data/scenarios/dev
python -m data.simulator --n 50 --split test --injected 10   # data/scenarios/test
python -m eval.run_experiment --scenarios data/scenarios/dev --preset mock --out results/dev_mock
python -m eval.run_experiment --scenarios data/scenarios/test --preset single_provider --seeds 42 43 44 --out results/run_01 --emissions
python -m eval.run_experiment --conditions graph,graph:no_ppr,graph:no_merge,graph:no_push,graph:no_mmr,graph:no_compression,graph:budget=100 --scenarios data/scenarios/dev --preset mock --out results/ablations
python -m viz.trace_view                  # trace of the worked example -> results/trace.html
python -m viz.trace_view --inject         # inject a wrong record; the trace points to it
python -m graphmem.warmup                 # download/cache heavy models
```

Each experiment directory gets `calls.jsonl`, `answers.jsonl`, `traces.jsonl`,
`memory_stats.jsonl`, `decisions.jsonl`, `per_run.csv`, `summary.csv`, `stats.csv`
(Wilcoxon vs graph) and `plots/`.

## Layout

```
graphmem/   memory system (compress, keywords, resolve, merge, notify, retrieve, trace, maintenance)
graphmem/graphs/   LangGraph write / read / maintenance graphs
agents/     prompts, schemas, model factory, mock LLM, callback logger, agent + orchestrator graphs
baselines/  full history, sliding window, rolling summary, vector RAG
data/       simulator, paraphraser, schema
eval/       experiment harness, metrics, statistics, plots
viz/        pyvis trace view
tests/      pytest suite (plan §20)
```
