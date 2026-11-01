"""Optional LLM rewording of scenario events (plan §14.1).

Ground truth is fixed by the simulator; a rewording is kept only if every number and
machine ID survives, so the correct answer can never change.

python -m data.paraphrase --in data/scenarios/dev --out data/scenarios/dev_para --preset single_provider
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Reword factory shop-floor messages naturally, as a technician would write them. "
               "Keep every number, unit, time and machine ID exactly. Output only the reworded message."),
    ("human", "{text}"),
])
NUM = re.compile(r"\d+(?:\.\d+)?")
MACHINE = re.compile(r"\b([A-Za-z]{2,5})[\s\-]?(\d{1,4})\b")


def facts(text: str) -> tuple[set, set]:
    return set(NUM.findall(text)), {f"{a.lower()}-{int(b)}" for a, b in MACHINE.findall(text)}


def paraphrase_scenario(sc: dict, chain) -> tuple[dict, int]:
    changed = 0
    for e in sc["events"]:
        if e["type"] == "question":
            continue
        out, _ = chain.invoke(PROMPT.format_messages(text=e["text"]))
        new = str(out.content).strip().strip('"')
        if new and facts(new) == facts(e["text"]):
            e["text"] = new
            changed += 1
    return sc, changed


def main():
    from agents.models import get_models
    from graphmem.config import load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="single_provider")
    args = ap.parse_args()
    cfg = load_config()
    chain = get_models(cfg, args.preset)["sensor"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(args.inp).glob("*.json")):
        sc, n = paraphrase_scenario(json.loads(f.read_text(encoding="utf-8")), chain)
        (out / f.name).write_text(json.dumps(sc, indent=1), encoding="utf-8")
        print(f"{f.name}: reworded {n} events")


if __name__ == "__main__":
    main()
