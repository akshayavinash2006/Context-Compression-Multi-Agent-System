"""Chat-model factory: one model chain per agent role (plan §13.6-13.7).

Each role gets a primary model (Ollama, Groq, Google Gemini, OpenRouter or the mock),
followed by fallbacks. Models are built with LangChain's `init_chat_model`, rate-limited
with `InMemoryRateLimiter`, and optionally cached with `SQLiteCache`.
"""
from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass, field

from langchain_core.rate_limiters import InMemoryRateLimiter

from agents.mock_llm import RuleBasedChatModel
from graphmem.config import resolve_path

log = logging.getLogger(__name__)

ROLES = ["sensor", "history", "diagnosis", "report", "summariser"]
KEY_ENV = {"groq": "GROQ_API_KEY", "google_genai": "GOOGLE_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
PACKAGE = {"ollama": "langchain_ollama", "groq": "langchain_groq", "google_genai": "langchain_google_genai",
           "openrouter": "langchain_openai"}


@dataclass
class ModelSpec:
    provider: str
    model: str
    llm: object

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass
class ModelChain:
    """Primary model plus fallbacks; `invoke` tries them in order and reports which one answered."""
    role: str
    specs: list[ModelSpec] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    primary_skipped: bool = False       # configured primary unavailable at build time

    def invoke(self, messages, tools=None, schema=None, config=None):
        errors = []
        for i0, spec in enumerate(self.specs):
            i = i0 + int(self.primary_skipped)
            llm = spec.llm
            try:
                if schema is not None:
                    runnable = llm.with_structured_output(schema)
                elif tools:
                    runnable = llm.bind_tools(tools)
                else:
                    runnable = llm
                cfg = dict(config or {})
                cfg["metadata"] = {**cfg.get("metadata", {}), "provider": spec.provider,
                                   "model": spec.model, "fallback_index": i}
                out = runnable.invoke(messages, config=cfg)
                return out, {"provider": spec.provider, "model": spec.model,
                             "fallback_used": i > 0, "errors": errors}
            except Exception as e:  # provider down, quota, unsupported feature...
                errors.append(f"{spec.label}: {type(e).__name__}: {str(e)[:160]}")
                log.warning("model %s failed for %s: %s", spec.label, self.role, e)
        raise RuntimeError(f"All models failed for role {self.role}: {errors}")

    @property
    def primary(self) -> str:
        return self.specs[0].label if self.specs else "none"


_ollama_ok: dict[str, bool] = {}


def ollama_reachable(base_url: str) -> bool:
    if base_url not in _ollama_ok:
        try:
            urllib.request.urlopen(base_url.rstrip("/") + "/api/tags", timeout=1.0)
            _ollama_ok[base_url] = True
        except Exception:
            _ollama_ok[base_url] = False
    return _ollama_ok[base_url]


def _package_ok(provider: str) -> bool:
    import importlib.util
    pkg = PACKAGE.get(provider)
    return pkg is None or importlib.util.find_spec(pkg) is not None


def build_llm(provider: str, model: str, cfg, seed: int | None = None):
    """Returns a LangChain chat model, or raises ValueError with the reason it is unavailable."""
    a = cfg.agents
    if provider == "mock":
        return RuleBasedChatModel(expand_policy=a.get("mock", {}).get("expand_policy", "short"))
    if not _package_ok(provider):
        raise ValueError(f"package {PACKAGE[provider]} not installed")
    if provider in KEY_ENV and not os.environ.get(KEY_ENV[provider]):
        raise ValueError(f"{KEY_ENV[provider]} not set")
    if provider == "ollama" and not ollama_reachable(a.ollama_base_url):
        raise ValueError(f"Ollama not reachable at {a.ollama_base_url}")
    rps = (a.get("rate_limits") or {}).get(provider, 0)
    limiter = InMemoryRateLimiter(requests_per_second=rps, check_every_n_seconds=0.1, max_bucket_size=1) \
        if rps else None
    from langchain.chat_models import init_chat_model  # noqa: WPS433 (optional heavy import)
    kwargs = {"temperature": a.temperature}
    if limiter:
        kwargs["rate_limiter"] = limiter
    if provider == "ollama":
        kwargs.update(base_url=a.ollama_base_url, num_ctx=a.num_ctx)
        if seed is not None:
            kwargs["seed"] = seed
        return init_chat_model(model, model_provider="ollama", **kwargs)
    if provider == "openrouter":
        return init_chat_model(model, model_provider="openai", base_url=a.openrouter_base_url,
                               api_key=os.environ[KEY_ENV["openrouter"]], **kwargs)
    return init_chat_model(model, model_provider=provider, **kwargs)


def get_models(cfg, preset: str | None = None, seed: int | None = None) -> dict[str, ModelChain]:
    """One ModelChain per role for the chosen preset (mock | single_provider | multi_provider)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(resolve_path(".env"))
    except ImportError:
        pass
    preset = preset or cfg.agents.preset
    table = cfg.agents.presets[preset]
    fallbacks = [] if preset == "mock" else list(cfg.agents.get("fallbacks") or [])
    chains = {}
    for role in ROLES:
        entries = [table.get(role) or table["default"], *fallbacks]
        chain = ModelChain(role)
        seen = set()
        for n, e in enumerate(entries):
            key = (e["provider"], e.get("model", ""))
            if key in seen:
                continue
            seen.add(key)
            try:
                llm = build_llm(e["provider"], e.get("model", ""), cfg, seed)
                chain.specs.append(ModelSpec(e["provider"], e.get("model", ""), llm))
            except Exception as err:
                chain.skipped.append(f"{e['provider']}:{e.get('model', '')} ({err})")
                if n == 0:
                    chain.primary_skipped = True
        if not chain.specs:
            chain.specs.append(ModelSpec("mock", "rule-based", build_llm("mock", "", cfg)))
            chain.skipped.append("no configured model available; using mock")
        chains[role] = chain
    return chains


def enable_llm_cache(cfg, preset: str) -> bool:
    """SQLite LLM cache so repeated runs are identical and cost no API calls."""
    path = cfg.runtime.get("llm_cache")
    if not path or preset == "mock":
        return False
    try:
        from langchain_community.cache import SQLiteCache
        from langchain_core.globals import set_llm_cache
        p = resolve_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        set_llm_cache(SQLiteCache(database_path=str(p)))
        return True
    except Exception as e:
        log.warning("LLM cache disabled: %s", e)
        return False


def describe(chains: dict[str, ModelChain]) -> str:
    rows = []
    for role, ch in chains.items():
        fb = " -> ".join(s.label for s in ch.specs[1:]) or "-"
        rows.append(f"  {role:<10} {ch.primary:<40} fallbacks: {fb}")
        for s in ch.skipped:
            rows.append(f"  {'':<10} skipped {s}")
    return "\n".join(rows)
