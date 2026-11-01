"""Download and cache all models once: python -m graphmem.warmup"""
from graphmem.config import load_config
from graphmem.memory import SharedModels


def main():
    cfg = load_config(overrides={"runtime.use_heavy_models": "auto"})
    m = SharedModels.get(cfg)
    print("embedder:  ", m["embedder"].kind)
    print("nli:       ", m["nli"].kind)
    print("compressor:", m["compressor"].backend_name)
    print("spaCy:     ", "loaded" if m["extractor"].nlp is not None else "fallback (regex + vocabulary)")
    print("KeyBERT:   ", "loaded" if m["extractor"].kb is not None else "fallback (YAKE)")
    print("tokens:    ", m["counter"].name)
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        print("Qwen tokenizer cached")
    except Exception as e:  # optional
        print("Qwen tokenizer not cached:", type(e).__name__)


if __name__ == "__main__":
    main()
