"""Metrics (plan §17) computed from the JSONL logs of one experiment directory."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from graphmem.models import normalize_machine

KEY = ["condition", "scenario", "seed"]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def action_score(pred: str, gold: str) -> float:
    p = set(re.findall(r"[a-z]+", (pred or "").lower()))
    g = set(re.findall(r"[a-z]+", (gold or "").lower()))
    return len(p & g) / len(g) if g else float(not p)


def canonical_events(scenario: dict) -> dict[str, str]:
    """Duplicates count as their original event (a merged memory carries both)."""
    return {e["id"]: e.get("duplicate_of") or e["id"] for e in scenario["events"]}


def run_metrics(key: tuple, calls, answers, turns, decisions, stats, runs, scenario) -> dict:
    r: dict = dict(zip(KEY, key))
    # ---- cost ----
    r["calls"] = len(calls)
    r["tokens_per_task"] = int(sum(c["input_tokens"] for c in calls))
    r["output_tokens"] = int(sum(c.get("output_tokens") or 0 for c in calls))
    r["summariser_tokens"] = int(sum(c["input_tokens"] for c in calls if c.get("agent") == "summariser"))
    if len(calls) >= 2:
        turns_ = np.array([c.get("turn", 0) for c in calls], float)
        toks = np.array([c["input_tokens"] for c in calls], float)
        r["growth_slope"] = float(np.polyfit(turns_, toks, 1)[0]) if np.ptp(turns_) > 0 else 0.0
    else:
        r["growth_slope"] = np.nan
    r["fallback_calls"] = int(sum(bool(c.get("fallback_used")) for c in calls))
    r["providers"] = ",".join(sorted({str(c.get("provider")) for c in calls}))
    # ---- accuracy ----
    acc, rc_acc, act, mh = [], [], [], []
    lat = []
    for a in answers:
        rep, gold = a.get("report") or {}, a.get("gold") or {}
        ok_rc = rep.get("root_cause") == gold.get("root_cause")
        ok = ok_rc and normalize_machine(rep.get("machine")) == normalize_machine(gold.get("machine"))
        acc.append(ok)
        rc_acc.append(ok_rc)
        act.append(action_score(rep.get("action", ""), gold.get("action", "")))
        if (a.get("hops") or 0) >= 2:
            mh.append(ok)
        lat.append(sum(c["latency_s"] for c in calls if c.get("event_id") == a["event_id"]
                       and c.get("agent") in ("diagnosis", "report")))
    r["accuracy"] = float(np.mean(acc)) if acc else np.nan
    r["root_cause_acc"] = float(np.mean(rc_acc)) if rc_acc else np.nan
    r["action_score"] = float(np.mean(act)) if act else np.nan
    r["multihop_acc"] = float(np.mean(mh)) if mh else np.nan
    r["latency_per_question_s"] = float(np.mean(lat)) if lat else np.nan
    # ---- context use ----
    agent_turns = [t for t in turns if t.get("delivered")]
    cited_turns = [t for t in agent_turns if not t.get("citation_missing")]
    r["citation_rate"] = len(cited_turns) / len(agent_turns) if agent_turns else np.nan
    r["unnecessary_context"] = float(np.mean([len(set(t["delivered"]) - set(t["cited"])) / len(t["delivered"])
                                              for t in cited_turns])) if cited_turns else np.nan
    r["context_tokens_per_turn"] = float(np.mean([t["context_tokens"] for t in turns])) if turns else np.nan
    r["expand_rate"] = sum(c.get("tool_calls", 0) for c in calls if c.get("agent") != "report") / max(1, len(calls))
    # ---- storage / compression ----
    final = next((s for s in stats if s.get("final")), {})
    r["active_memories"] = final.get("active_memories")
    r["stored_tokens"] = final.get("stored_tokens_active")
    r["tokens_written"] = final.get("orig_tokens_written")
    r["compression_ratio"] = final.get("compression_ratio")
    r["compress_fallback_rate"] = final.get("compress_fallbacks", 0) / max(1, final.get("writes", 1))
    r["wall_s"] = (runs[0] if runs else {}).get("wall_s")
    r["emissions_kg"] = (runs[0] if runs else {}).get("emissions_kg")
    # ---- traceability ----
    canon = canonical_events(scenario)
    tp_ = fp_ = fn_ = 0
    for a in answers:
        gold = set((a.get("gold") or {}).get("supporting_events") or [])
        if not gold:
            continue
        leaves = {canon.get(e, e) for e in a.get("trace_leaves", [])}
        gold = {canon.get(e, e) for e in gold}
        tp_ += len(leaves & gold)
        fp_ += len(leaves - gold)
        fn_ += len(gold - leaves)
    r["trace_precision"] = tp_ / (tp_ + fp_) if tp_ + fp_ else np.nan
    r["trace_recall"] = tp_ / (tp_ + fn_) if tp_ + fn_ else np.nan
    inj = (scenario.get("injected_error") or {}).get("event_id")
    if inj:
        r["trace_finds_injected"] = float(any(inj in a.get("trace_leaves", []) for a in answers))
        r["items_inspected_trace"] = float(np.mean([a.get("trace_size", 0) for a in answers])) if answers else np.nan
        r["items_inspected_history"] = float(np.mean([a.get("history_size", 0) for a in answers])) if answers else np.nan
    # ---- merge / update / routing (graph memory only) ----
    if decisions:
        r.update(merge_routing_metrics(decisions, scenario))
    return r


def merge_routing_metrics(decisions: list[dict], scenario: dict) -> dict:
    events = {e["id"]: e for e in scenario["events"]}
    by_event = {d["event_id"]: d for d in decisions if d.get("event_id") in events
                and events[d["event_id"]]["type"] != "question"}
    mem_of = {eid: d["memory_id"] for eid, d in by_event.items()}
    out = {}
    gold_dup = {eid: e["duplicate_of"] for eid, e in events.items() if e.get("duplicate_of")}
    merges = [eid for eid, d in by_event.items() if d["decision"] == "merge"]
    hit = [eid for eid in gold_dup if by_event.get(eid, {}).get("decision") == "merge"
           and by_event[eid]["memory_id"] == mem_of.get(gold_dup[eid])]
    out["merge_precision"] = len([e for e in merges if e in hit]) / len(merges) if merges else np.nan
    out["merge_recall"] = len(hit) / len(gold_dup) if gold_dup else np.nan
    gold_cor = {eid: e["corrects"] for eid, e in events.items() if e.get("corrects")}
    updates = [eid for eid, d in by_event.items() if d["decision"] == "update"]
    hit_u = [eid for eid in gold_cor if by_event.get(eid, {}).get("decision") == "update"
             and by_event[eid].get("superseded") == mem_of.get(gold_cor[eid])]
    out["update_precision"] = len([e for e in updates if e in hit_u]) / len(updates) if updates else np.nan
    out["update_recall"] = len(hit_u) / len(gold_cor) if gold_cor else np.nan
    tp = fp = fn = 0
    for eid, d in by_event.items():
        e = events[eid]
        gold = set(e.get("needed_by") or []) - {e["source_agent"]}
        pred = set(d.get("notified") or [])
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
    out["routing_precision"] = tp / (tp + fp) if tp + fp else np.nan
    out["routing_recall"] = tp / (tp + fn) if tp + fn else np.nan
    return out


def group(rows: list[dict]) -> dict[tuple, list[dict]]:
    g: dict[tuple, list[dict]] = {}
    for r in rows:
        g.setdefault(tuple(r[k] for k in KEY), []).append(r)
    return g


def per_run_table(out: Path) -> pd.DataFrame:
    scenarios = json.loads((out / "scenarios.json").read_text(encoding="utf-8"))
    logs = {n: group(read_jsonl(out / f"{n}.jsonl"))
            for n in ("calls", "answers", "turns", "decisions", "memory_stats", "runs")}
    keys = sorted(set(logs["runs"]) | set(logs["answers"]))
    rows = []
    for k in keys:
        rows.append(run_metrics(k, logs["calls"].get(k, []), logs["answers"].get(k, []),
                                logs["turns"].get(k, []), logs["decisions"].get(k, []),
                                logs["memory_stats"].get(k, []), logs["runs"].get(k, []), scenarios[k[1]]))
    return pd.DataFrame(rows)


SUMMARY_COLS = ["tokens_per_task", "growth_slope", "accuracy", "multihop_acc", "action_score",
                "latency_per_question_s", "unnecessary_context", "citation_rate", "expand_rate",
                "context_tokens_per_turn", "stored_tokens", "compression_ratio", "trace_precision",
                "trace_recall", "merge_precision", "merge_recall", "update_precision", "update_recall",
                "routing_precision", "routing_recall", "emissions_kg"]


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in SUMMARY_COLS if c in df and df[c].notna().any()]
    agg = df.groupby("condition")[cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg


def compute_all(out: str | Path) -> pd.DataFrame:
    out = Path(out)
    df = per_run_table(out)
    df.to_csv(out / "per_run.csv", index=False)
    summ = summarise(df)
    summ.to_csv(out / "summary.csv")
    try:
        from eval.stats import compare
        compare(df).to_csv(out / "stats.csv", index=False)
    except Exception as e:  # too few runs for a test
        (out / "stats.csv").write_text(f"not computed: {e}\n", encoding="utf-8")
    view = [c for c in ["tokens_per_task_mean", "accuracy_mean", "multihop_acc_mean", "unnecessary_context_mean",
                        "trace_precision_mean", "trace_recall_mean", "stored_tokens_mean"] if c in summ]
    return summ[view].round(3)
