"""Check API keys and list available models: python check_apis.py

Prints which keys are set, the chat models each provider offers, and whether the
models configured in config.yaml (multi_provider preset) answer a one-line test.
Keys are never printed.
"""
import json
import os
import urllib.error
import urllib.request

from dotenv import load_dotenv

from graphmem.config import load_config, resolve_path

load_dotenv(resolve_path(".env"))


def get(url, headers=None):
    # Groq's API is behind Cloudflare, which rejects the default "Python-urllib" user agent (403).
    req = urllib.request.Request(url, headers={"User-Agent": "graphmem-check/1.0",
                                               "Accept": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def http_error(e) -> str:
    """Short, key-free description of an HTTP error."""
    if isinstance(e, urllib.error.HTTPError):
        try:
            body = e.read().decode("utf-8", "replace")[:200]
        except Exception:
            body = ""
        return f"HTTP {e.code} {e.reason} {body}".strip()
    return type(e).__name__


def main():
    groq, google = os.environ.get("GROQ_API_KEY", ""), os.environ.get("GOOGLE_API_KEY", "")
    print("GROQ_API_KEY:  ", "set" if groq else "MISSING")
    print("GOOGLE_API_KEY:", "set" if google else "MISSING")

    if groq:
        print("\nListing Groq models...", flush=True)
        try:
            data = get("https://api.groq.com/openai/v1/models", {"Authorization": f"Bearer {groq}"})
            names = sorted(m["id"] for m in data["data"] if "whisper" not in m["id"] and "tts" not in m["id"])
            print("Groq models:\n  " + "\n  ".join(names))
        except Exception as e:
            print("Groq: could not list models:", http_error(e))
    if google:
        print("\nListing Gemini models...", flush=True)
        try:
            data = get("https://generativelanguage.googleapis.com/v1beta/models",
                       {"x-goog-api-key": google})
            names = sorted(m["name"].split("/")[-1] for m in data.get("models", [])
                           if "generateContent" in m.get("supportedGenerationMethods", []))
            print("Gemini models:\n  " + "\n  ".join(names))
        except Exception as e:
            print("Gemini: could not list models:", http_error(e))

    print("\nLoading project...", flush=True)
    from agents.models import build_llm
    cfg = load_config()
    table = cfg.agents.presets.multi_provider
    print("Testing configured models (one short call each):", flush=True)
    for role in ("diagnosis", "report"):
        e = table[role]
        try:
            out = build_llm(e["provider"], e["model"], cfg).invoke("Reply with the single word: ok")
            print(f"  {role:<10} {e['provider']}:{e['model']:<35} OK -> {str(out.content)[:40]!r}")
        except Exception as err:
            print(f"  {role:<10} {e['provider']}:{e['model']:<35} FAILED: {str(err)[:600]}")


if __name__ == "__main__":
    main()
