"""Deterministic scenario generator with ground truth (plan §14).

python -m data.simulator --n 10 --split dev --out data/scenarios/dev
python -m data.simulator --n 50 --split test --injected 10 --out data/scenarios/test
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from graphmem.models import ACTIONS

MACHINE_POOL = [f"cnc-{i}" for i in range(1, 10)] + [f"vmc-{i}" for i in range(10, 15)] + \
               [f"lathe-{i}" for i in range(2, 6)]
VIB_LIMIT, TEMP_LIMIT = 6, 70

CAUSES = {
    "bearing_wear": {
        "symptom": "vibration",
        "sensor": ["{M} spindle vibration hit {v} mm/s at {time}, above the {lim} mm/s limit.",
                   "Hey team, {M} spindle vibration reached {v} mm/s at {time}, which is above the "
                   "{lim} mm/s limit. Temperature looks fine."],
        "history": ["{M} spindle bearing was last replaced {age} months ago; the service interval is {iv} months.",
                    "Maintenance log: spindle bearing on {M} replaced {age} months ago, interval {iv} months."],
        "extra": ["Operator reports a grinding noise from the {M} spindle during finishing passes."],
    },
    "misalignment": {
        "symptom": "vibration",
        "sensor": ["Axial vibration on {M} rose to {v} mm/s at {time}, above the {lim} mm/s limit.",
                   "{M} axial vibration is {v} mm/s at {time}, above the {lim} mm/s limit."],
        "history": ["{M} motor coupling was refitted {d} days ago without an alignment check.",
                    "Maintenance log: coupling on {M} reinstalled {d} days ago; alignment was not verified."],
        "extra": ["Operator notes {M} vibration got worse right after the coupling work."],
    },
    "imbalance": {
        "symptom": "vibration",
        "sensor": ["{M} spindle vibration {v} mm/s at 1x running speed at {time}, above the {lim} mm/s limit.",
                   "{M} vibration at running speed is {v} mm/s at {time}, above the {lim} mm/s limit."],
        "history": ["New fixture on {M} was mounted off-centre during setup {d} days ago.",
                    "Setup record: {M} fixture installed off-centre {d} days ago."],
        "extra": ["Operator says {M} vibration started when the new fixture went on."],
    },
    "coolant_failure": {
        "symptom": "temperature",
        "sensor": ["{M} spindle temperature reached {t} °C at {time}, above the {tlim} °C limit.",
                   "Hey team, {M} spindle temperature is {t} °C at {time}, above the {tlim} °C limit."],
        "history": ["{M} coolant pump flow dropped to {p}% of rated flow last week.",
                    "Maintenance log: coolant pump on {M} delivering only {p}% of rated flow."],
        "extra": ["Operator notes coolant spray on {M} looks weak."],
    },
    "overload": {
        "symptom": "current",
        "sensor": ["{M} motor current {c} A at {time}, above the rated {r} A.",
                   "{M} spindle motor drawing {c} A at {time}, above the rated {r} A."],
        "history": ["Operator note: {M} is running a heavy roughing job with deep cuts all shift.",
                    "Production plan: {M} assigned heavy roughing with deep cuts for job {job}."],
        "extra": ["{M} cycle time increased by {x}% on the roughing job."],
    },
    "lubrication_failure": {
        "symptom": "temperature",
        "sensor": ["{M} ball screw temperature {t} °C at {time} and rising, above the {tlim} °C limit.",
                   "{M} ball screw is at {t} °C at {time}, above the {tlim} °C limit and rising."],
        "history": ["{M} lubrication unit reported low oil level; last refill {d} days ago.",
                    "Maintenance log: oil level low on {M} lubrication unit, last refill {d} days ago."],
        "extra": ["Operator hears a squeal from the {M} ball screw."],
    },
    "sensor_fault": {
        "symptom": "vibration",
        "sensor": ["{M} vibration reading jumped to {v} mm/s at {time} and back to 0.0 mm/s within one second.",
                   "{M} vibration value jumped to {v} mm/s at {time}, then 0.0 mm/s a second later."],
        "history": ["{M} vibration sensor was flagged for calibration {d} days ago.",
                    "Maintenance log: calibration of the {M} vibration sensor is pending since {d} days."],
        "extra": ["Operator says {M} sounds and feels normal."],
    },
    "belt_wear": {
        "symptom": "speed",
        "sensor": ["{M} spindle speed fluctuating by {x}% under load at {time}.",
                   "{M} spindle speed is fluctuating {x}% under load at {time}."],
        "history": ["{M} drive belt shows cracks; it was installed {age} months ago, interval {iv} months.",
                    "Inspection: cracks on the {M} drive belt, installed {age} months ago (interval {iv} months)."],
        "extra": ["Operator hears a slapping noise from the {M} belt guard."],
    },
}
QUESTIONS = {
    "vibration": ["Why is {M} vibrating and what should we do?", "What is causing the vibration on {M}?"],
    "temperature": ["Why is {M} running hot and what should we do?", "What is causing the high temperature on {M}?"],
    "current": ["Why is {M} drawing high current and what should we do?"],
    "speed": ["Why is the {M} spindle speed unstable and what should we do?"],
}
DUPLICATE = {
    "vibration": "Second report on {M}: vibration measured {v} mm/s at {time}, over the {lim} mm/s limit.",
    "temperature": "Second report on {M}: temperature measured {t} °C at {time}, over the {tlim} °C limit.",
    "current": "Second report on {M}: motor current measured {c} A at {time}, over the rated {r} A.",
    "speed": "Second report on {M}: spindle speed fluctuating {x}% under load at {time}.",
}
DISTRACTORS = [
    ("sensor_reading", "sensor", "{M} spindle vibration {v2} mm/s at {time}, within the {lim} mm/s limit."),
    ("maintenance", "history", "{M} spindle bearing replaced {a2} months ago; the service interval is {iv} months."),
    ("maintenance", "history", "{M} coolant pump flow normal at 100% of rated flow."),
    ("sensor_reading", "sensor", "{M} motor current {c2} A at {time}, within the rated {r} A."),
    ("sensor_reading", "sensor", "{M} spindle temperature {t2} °C at {time}, normal."),
    ("maintenance", "history", "{M} drive belt inspected {d} days ago, condition normal."),
]
FILLERS = [
    ("operator_note", "sensor", "Shift handover: {M} running batch {job}, no issues reported."),
    ("operator_note", "sensor", "{M} tool change completed for job {job}."),
    ("operator_note", "sensor", "{M} completed batch {job} of flange parts."),
    ("maintenance", "history", "{M} hydraulic pressure normal at {bar} bar after the weekly check."),
    ("operator_note", "sensor", "Material delivery for job {job} arrived at the stores."),
    ("maintenance", "history", "Supplier delivery of spare spindle bearings delayed by {d} days."),
]
CHATTER_PRE = ["Hey team, ", "FYI: ", "Quick update from the floor: ", "Just a heads-up, ", ""]
CHATTER_POST = [" Let me know if you need more details.", " I will keep an eye on it for the rest of the shift.",
                " Please check when you get a chance.", " This was also written in the shift book.", ""]
SPECS = [
    ("Vibration limit for all CNC, VMC and lathe machines is {lim} mm/s.", ["sensor", "diagnosis"]),
    ("Spindle and ball screw temperature limit is {tlim} °C.", ["sensor", "diagnosis"]),
    ("Rated spindle motor current is {r} A on all machines.", ["sensor", "diagnosis"]),
]


def surface(machine: str, rng: random.Random) -> str:
    p, n = machine.split("-")
    return rng.choices([f"{p.upper()}-{n}", f"{p.upper()} {n}", f"{p.upper()}{n}"], weights=[6, 2, 1])[0]


def values(rng: random.Random) -> dict:
    iv = rng.choice([12, 18, 24])
    return {
        "v": round(VIB_LIMIT + rng.uniform(0.8, 3.5), 1), "v2": round(rng.uniform(1.5, 4.5), 1),
        "lim": VIB_LIMIT, "t": rng.randint(76, 92), "t2": rng.randint(40, 60), "tlim": TEMP_LIMIT,
        "c": rng.randint(34, 45), "c2": rng.randint(15, 26), "r": 30, "p": rng.randint(35, 65),
        "iv": iv, "age": iv + rng.randint(2, 6), "a2": rng.randint(1, iv - 4), "d": rng.randint(2, 20),
        "x": rng.randint(8, 20), "job": rng.randint(1000, 9999), "bar": rng.randint(55, 70),
        "time": f"{rng.randint(6, 21):02d}:{rng.choice([0, 5, 10, 20, 35, 45, 50]):02d}",
    }


def alter_number(text: str, rng: random.Random) -> str:
    """Wrong version of a fact: first plain number replaced (for correction events)."""
    m = re.search(r"(?<![\d.:])(\d+)(?![\d.:])", text)
    if not m:
        return text
    n = int(m.group(1))
    wrong = max(1, n - rng.randint(3, 8)) if n > 8 else n + rng.randint(3, 8)
    return text[:m.start()] + str(wrong) + text[m.end():]


def generate(idx: int, seed: int = 42, split: str = "test", injected: bool = False) -> dict:
    rng = random.Random(seed * 1000 + idx)
    k = rng.randint(4, 8)
    machines = rng.sample(MACHINE_POOL, k)
    n_q = rng.randint(3, min(6, k))
    with_none = rng.random() < 0.3
    n_faults = n_q - (1 if with_none else 0)
    causes = rng.sample(list(CAUSES), n_faults)
    faulty = machines[:n_faults]
    healthy = machines[n_faults:]
    total = rng.randint(40, 80)
    base_vals = values(rng)

    slots: dict[int, dict] = {}
    q_positions = sorted(rng.sample(range(int(total * 0.45), total), n_q))
    free = [p for p in range(3, total) if p not in q_positions]

    def take(lo: int, hi: int) -> int | None:
        cands = [p for p in free if lo <= p < hi]
        if not cands:
            return None
        p = rng.choice(cands)
        free.remove(p)
        return p

    for i, (text, needed) in enumerate(SPECS):                       # specs first
        slots[i] = {"type": "spec", "source_agent": "history", "text": text.format(**base_vals),
                    "entities": [], "needed_by": needed}
    for p in range(len(SPECS)):
        if p in free:
            free.remove(p)

    questions = []
    order = list(range(n_q))
    rng.shuffle(order)
    for qi, pos in zip(order, q_positions):
        if qi < n_faults:
            questions.append((pos, faulty[qi], causes[qi]))
        else:
            questions.append((pos, healthy[0] if healthy else machines[-1], "none"))

    injected_error = None
    for pos, machine, cause in questions:
        v = {**base_vals, **values(rng)}
        M = surface(machine, rng)
        supporting = []
        if cause == "none":
            q_text = f"Is anything wrong with {M} and does it need maintenance?"
            gold = {"machine": machine, "root_cause": "none", "action": ACTIONS["none"], "supporting_events": []}
            slots[pos] = {"type": "question", "source_agent": "user", "text": q_text, "gold": gold, "hops": 0}
            continue
        spec = CAUSES[cause]
        sym = spec["symptom"]
        s_text = rng.choice(spec["sensor"]).format(M=M, **v)
        h_text = rng.choice(spec["history"]).format(M=surface(machine, rng), **v)
        p_s = take(len(SPECS), pos)
        p_h = take(len(SPECS), pos)
        if p_s is None or p_h is None:
            continue
        slots[p_s] = {"type": "sensor_reading", "source_agent": "sensor", "text": s_text,
                      "entities": [machine, sym], "needed_by": ["diagnosis"], "role": "sensor"}
        h_type = "operator_note" if cause == "overload" and h_text.startswith("Operator") else "maintenance"
        h_src = "sensor" if h_type == "operator_note" else "history"
        slots[p_h] = {"type": h_type, "source_agent": h_src, "text": h_text,
                      "entities": [machine], "needed_by": ["diagnosis"], "role": "history"}
        supporting = [p_s, p_h]
        if rng.random() < 0.35:                                      # 3-hop: extra confirming fact
            p_x = take(len(SPECS), pos)
            if p_x is not None:
                slots[p_x] = {"type": "operator_note", "source_agent": "sensor",
                              "text": rng.choice(spec["extra"]).format(M=surface(machine, rng), **v),
                              "entities": [machine], "needed_by": ["diagnosis"], "role": "extra"}
                supporting.append(p_x)
        if rng.random() < 0.5:                                       # duplicate of the sensor fact
            p_d = take(p_s + 1, pos)
            if p_d is not None:
                slots[p_d] = {"type": "operator_note", "source_agent": "sensor",
                              "text": DUPLICATE[sym].format(M=surface(machine, rng), **v),
                              "entities": [machine, sym], "needed_by": ["diagnosis"], "dup_of_pos": p_s}
        if rng.random() < 0.35:                                      # correction of the history fact
            p_w = take(len(SPECS), p_h)
            if p_w is not None and p_w < p_h:
                slots[p_w] = {"type": "maintenance", "source_agent": "history", "text": alter_number(h_text, rng),
                              "entities": [machine], "needed_by": [], "role": "wrong"}
                slots[p_h]["text"] = "Correction: " + h_text + " The earlier record was wrong."
                slots[p_h]["corrects_pos"] = p_w
            elif p_w is not None:
                free.append(p_w)
        if injected and injected_error is None:                      # debugging study
            wrong_cause = rng.choice([c for c in CAUSES if c != cause and CAUSES[c]["symptom"] == sym]
                                     or [c for c in CAUSES if c != cause])
            slots[p_h]["text"] = rng.choice(CAUSES[wrong_cause]["history"]).format(M=surface(machine, rng), **v)
            slots[p_h].pop("corrects_pos", None)
            injected_error = {"pos": p_h, "description": f"history fact replaced by a fact pointing to {wrong_cause}",
                              "wrong_cause": wrong_cause}
        q_text = rng.choice(QUESTIONS[sym]).format(M=surface(machine, rng))
        slots[pos] = {"type": "question", "source_agent": "user", "text": q_text,
                      "gold": {"machine": machine, "root_cause": cause, "action": ACTIONS[cause],
                               "supporting_pos": supporting},
                      "hops": len(supporting)}

    # distractors (other machines with overlapping words) and fillers
    others = healthy or machines
    for p in sorted(free):
        v = {**base_vals, **values(rng)}
        if rng.random() < 0.45:
            typ, src, tmpl = rng.choice(DISTRACTORS)
            m = rng.choice(others)
        else:
            typ, src, tmpl = rng.choice(FILLERS)
            m = rng.choice(machines)
        slots[p] = {"type": typ, "source_agent": src, "text": tmpl.format(M=surface(m, rng), **v),
                    "entities": [m], "needed_by": []}

    # informal, noisy wording on some messages (plan §14.2 operator notes)
    for p, e in slots.items():
        if e["type"] not in ("question", "spec") and rng.random() < 0.4:
            e["text"] = rng.choice(CHATTER_PRE) + e["text"] + rng.choice(CHATTER_POST)

    # assign ids in turn order and resolve position references
    pos_to_id = {}
    events = []
    for n, p in enumerate(sorted(slots), start=1):
        pos_to_id[p] = f"E{n:03d}"
    for p in sorted(slots):
        e = slots[p]
        ev = {"id": pos_to_id[p], "turn": len(events) + 1, "type": e["type"], "source_agent": e["source_agent"],
              "text": e["text"]}
        if e["type"] == "question":
            g = dict(e["gold"])
            g["supporting_events"] = [pos_to_id[x] for x in g.pop("supporting_pos", [])] if "supporting_pos" in g \
                else g.get("supporting_events", [])
            ev["gold"] = g
            ev["hops"] = e["hops"]
        else:
            ev["entities"] = e.get("entities", [])
            ev["needed_by"] = e.get("needed_by", [])
            ev["duplicate_of"] = pos_to_id.get(e["dup_of_pos"]) if "dup_of_pos" in e else None
            ev["corrects"] = pos_to_id.get(e["corrects_pos"]) if "corrects_pos" in e else None
        events.append(ev)
    if injected_error:
        injected_error = {"event_id": pos_to_id[injected_error.pop("pos")], **injected_error}
    return {"scenario_id": f"{split[0].upper()}{idx:03d}", "split": split, "machines": machines,
            "events": events, "injected_error": injected_error}


def worked_example() -> dict:
    """The §2.3 worked example as a tiny scenario."""
    ev = [
        ("spec", "history", "Vibration limit for CNC machines is 6 mm/s."),
        ("operator_note", "sensor", "CNC-5 completed batch 4411 of flange parts."),
        ("sensor_reading", "sensor", "Hey team, CNC-3 spindle vibration hit 7.2 mm/s at 14:05, above the 6 mm/s limit."),
        ("operator_note", "sensor", "Material delivery for job 5120 arrived at the stores."),
        ("maintenance", "history", "CNC-3 spindle bearing was last replaced 14 months ago; the service interval is 12 months."),
        ("sensor_reading", "sensor", "CNC-5 spindle vibration 3.1 mm/s at 14:10, within the 6 mm/s limit, normal."),
        ("operator_note", "sensor", "Second report on CNC-3: vibration measured 7.2 mm/s at 14:05, over the 6 mm/s limit."),
        ("operator_note", "sensor", "Shift handover: VMC-12 running batch 2210, no issues reported."),
        ("question", "user", "Why is CNC-3 vibrating and what should we do?"),
    ]
    events = []
    for i, (t, src, text) in enumerate(ev, start=1):
        e = {"id": f"E{i:03d}", "turn": i, "type": t, "source_agent": src, "text": text}
        if t == "question":
            e["gold"] = {"machine": "cnc-3", "root_cause": "bearing_wear", "action": ACTIONS["bearing_wear"],
                         "supporting_events": ["E003", "E005"]}
            e["hops"] = 2
        else:
            e.update(entities=[], needed_by=["diagnosis"] if i in (3, 5, 7) else [],
                     duplicate_of="E003" if i == 7 else None, corrects=None)
        events.append(e)
    return {"scenario_id": "WORKED", "split": "demo", "machines": ["cnc-3", "cnc-5", "vmc-12"],
            "events": events, "injected_error": None}


def validate(s: dict) -> None:
    """Minimal schema check (data/schema.json)."""
    assert {"scenario_id", "machines", "events"} <= set(s)
    ids = set()
    for e in s["events"]:
        assert {"id", "turn", "type", "source_agent", "text"} <= set(e), e
        assert e["type"] in {"spec", "sensor_reading", "maintenance", "operator_note", "question"}
        ids.add(e["id"])
        if e["type"] == "question":
            assert e["gold"]["root_cause"] in ACTIONS
            assert set(e["gold"]["supporting_events"]) <= ids, "supporting events must come before the question"
        else:
            for ref in ("duplicate_of", "corrects"):
                assert e.get(ref) is None or e[ref] in ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--injected", type=int, default=0, help="extra scenarios with an injected wrong fact")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or f"data/scenarios/{args.split}")
    out.mkdir(parents=True, exist_ok=True)
    start = 1 if args.split == "dev" else 101
    for i in range(args.n):
        s = generate(start + i, args.seed, args.split)
        validate(s)
        (out / f"{s['scenario_id']}.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    for i in range(args.injected):
        s = generate(900 + i, args.seed, args.split, injected=True)
        s["scenario_id"] = f"X{i + 1:03d}"
        validate(s)
        (out / f"{s['scenario_id']}.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    print(f"wrote {args.n + args.injected} scenarios to {out}")


if __name__ == "__main__":
    main()
