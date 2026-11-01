"""Interactive trace view (plan §19): answer -> facts -> agents, exported with pyvis.

python -m viz.trace_view                       # worked example, mock models
python -m viz.trace_view --scenario data/scenarios/dev/D001.json --question 1 --out results/trace.html
python -m viz.trace_view --inject             # demo: inject a wrong reading, trace points to it
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

COLORS = {"agent": "#9e9e9e", "observation": "#7e57c2", "derived": "#ffb300", "superseded": "#e57373"}


def trace_to_html(tr: dict, path: str | Path, title: str = "Trace") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nodes, edges = {}, []

    def walk(n):
        kind = "superseded" if n.get("superseded") else n["kind"]
        originals = "\n".join(f"[{o['agent']} t={o['turn']}] {o['text']}" for o in n.get("originals", []))
        nodes[n["id"]] = {"label": n["id"], "color": COLORS.get(kind, "#7e57c2"),
                          "title": f"{n['text']}\n\nORIGINAL:\n{originals}", "shape": "box"}
        for a in n.get("agents", []):
            aid = f"A:{a}"
            nodes.setdefault(aid, {"label": a, "color": COLORS["agent"], "title": f"agent {a}", "shape": "ellipse"})
            edges.append((n["id"], aid, "written_by"))
        for s in n.get("sources", []):
            edges.append((n["id"], s["id"], "derived_from"))
            walk(s)
    walk(tr)
    try:
        from pyvis.network import Network
        net = Network(height="650px", width="100%", directed=True, notebook=False, cdn_resources="in_line")
        for nid, d in nodes.items():
            net.add_node(nid, **d)
        for u, v, label in edges:
            net.add_edge(u, v, label=label, dashes=label == "written_by")
        net.set_options(json.dumps({"layout": {"hierarchical": {"enabled": True, "direction": "UD",
                                                                 "sortMethod": "directed"}},
                                    "physics": {"enabled": False}}))
        path.write_text(net.generate_html(), encoding="utf-8")
    except ImportError:  # plain HTML fallback
        rows = "".join(f"<li><b>{html.escape(n)}</b>: {html.escape(d['title'])}</li>" for n, d in nodes.items())
        path.write_text(f"<!doctype html><title>{html.escape(title)}</title><ul>{rows}</ul>", encoding="utf-8")
    return path


def main():
    from agents.callbacks import CallLogger
    from agents.graph import run_scenario
    from agents.models import get_models
    from data.simulator import worked_example
    from graphmem.config import load_config
    from graphmem.memory import GraphMemory

    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--question", type=int, default=0, help="index of the question to trace")
    ap.add_argument("--preset", default="mock")
    ap.add_argument("--inject", action="store_true", help="replace the bearing record with a wrong fact")
    ap.add_argument("--out", default="results/trace.html")
    ap.add_argument("--graphml", default="results/memory_graph.graphml")
    args = ap.parse_args()

    cfg = load_config()
    sc = json.loads(Path(args.scenario).read_text(encoding="utf-8")) if args.scenario else worked_example()
    if args.inject:
        for e in sc["events"]:
            if "bearing" in e["text"] and e["type"] == "maintenance":
                e["text"] = e["text"].split(" spindle")[0] + " motor coupling was refitted 3 days ago without an alignment check."
                sc["injected_error"] = {"event_id": e["id"]}
                break
    mem = GraphMemory(cfg, scenario_id=sc["scenario_id"])
    out = run_scenario(sc, mem, get_models(cfg, args.preset), CallLogger(), cfg)
    a = out["answers"][args.question]
    tr = mem.trace(a["report_memory_id"])
    p = trace_to_html(tr, args.out, a["question"])
    Path(args.graphml).parent.mkdir(parents=True, exist_ok=True)
    mem.export_graph(args.graphml)
    print("question:", a["question"])
    print("report:  ", json.dumps(a["report"]))
    print("gold:    ", a["gold"]["root_cause"])
    from graphmem.trace import iter_nodes
    for n in iter_nodes(tr):
        print(f"  {n['id']:<4} {n['kind']:<11} by {','.join(n['agents']):<10} {n['text'][:90]}")
        for o in n["originals"]:
            print(f"        original ({o['event_id']}): {o['text'][:90]}")
    if sc.get("injected_error"):
        hit = sc["injected_error"]["event_id"] in {o["event_id"] for n in iter_nodes(tr) for o in n["originals"]}
        print("trace contains the injected wrong fact:", hit)
    print(f"trace HTML: {p}\ngraph GraphML: {args.graphml}")


if __name__ == "__main__":
    main()
