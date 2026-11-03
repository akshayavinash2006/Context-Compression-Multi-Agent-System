"""Car plant scenarios: each production zone has its own monitoring agent.

python -m data.automotive --n 10 --out data/scenarios/plant

Zones and their machines (machine IDs use the prefixes in domain/vocabulary.yaml):
  press       Press Shop             prs-1, prs-2     stamping presses
  body        Body Shop (Welding)    rob-11, rob-12   welding robots
  paint       Paint Shop             pnt-3, pnt-4     paint booths / ovens
  powertrain  Powertrain             cnc-5, cnc-6     engine & gearbox machining
  assembly    Final Assembly         asm-7, agv-8     assembly line, AGVs
  quality     Quality                qc-9             end-of-line test rig
Support agents: history (Maintenance office), diagnosis (Control Room), report (Reports office).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.simulator import generate, validate

ZONES = {
    "press":      {"name": "Press Shop", "machines": ["prs-1", "prs-2"],
                   "description": "Stamping presses that form body panels from steel sheet."},
    "body":       {"name": "Body Shop", "machines": ["rob-11", "rob-12"],
                   "description": "Welding robots that join panels into the car body."},
    "paint":      {"name": "Paint Shop", "machines": ["pnt-3", "pnt-4"],
                   "description": "Paint booths and curing ovens."},
    "powertrain": {"name": "Powertrain", "machines": ["cnc-5", "cnc-6"],
                   "description": "CNC machining of engine blocks and gearbox parts."},
    "assembly":   {"name": "Final Assembly", "machines": ["asm-7", "agv-8"],
                   "description": "Assembly line and automated guided vehicles."},
    "quality":    {"name": "Quality", "machines": ["qc-9"],
                   "description": "End-of-line test rig and inspection."},
}
SUPPORT = {
    "history":   {"name": "Maintenance", "description": "Maintenance logs, service intervals and specs."},
    "diagnosis": {"name": "Control Room", "description": "Finds root causes from sensor and maintenance facts."},
    "report":    {"name": "Reports", "description": "Writes the final structured report."},
}
POOL = [m for z in ZONES.values() for m in z["machines"]]
ZONE_OF = {m: zid for zid, z in ZONES.items() for m in z["machines"]}
ZONE_SOURCES = {"sensor_reading", "operator_note"}       # shop-floor messages come from the zone agent


def generate_plant(idx: int, seed: int = 42, split: str = "plant", injected: bool = False) -> dict:
    s = generate(idx, seed, split, injected, pool=POOL, n_machines=(6, 9))
    s["scenario_id"] = f"P{idx:03d}" if not injected else s["scenario_id"]
    s["zones"] = ZONES
    s["support_agents"] = SUPPORT
    for ev in s["events"]:
        machine = next((e for e in ev.get("entities", []) if e in ZONE_OF), None)
        if machine is None and ev["type"] == "question":
            machine = ev["gold"]["machine"]
        ev["zone"] = ZONE_OF.get(machine)
        if ev["type"] in ZONE_SOURCES and ev["zone"]:
            ev["source_agent"] = ev["zone"]
            ev["needed_by"] = ev.get("needed_by", [])
    return s


def zone_agents(scenario: dict) -> dict[str, dict]:
    """Zone agents of a scenario (empty for the original single-sensor scenarios)."""
    return scenario.get("zones") or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/scenarios/plant")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(1, args.n + 1):
        s = generate_plant(i, args.seed)
        validate(s)
        (out / f"{s['scenario_id']}.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    print(f"wrote {args.n} car-plant scenarios to {out}")


if __name__ == "__main__":
    main()
