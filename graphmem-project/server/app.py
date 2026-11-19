"""FastAPI server for the 3D factory UI.

    python -m server                 # http://localhost:8000  (serves ui/dist if it has been built)

WebSocket /ws/run
  client -> {"type": "start", "scenario": "P001", "preset": "mock", "condition": "graph",
             "mode": "replay", "delay": 0.6, "budget": 300}
  client -> {"type": "stop"}
  server -> {"type": "run_started" | "event" | "write" | "read" | "llm_start" | "llm" |
             "maintain" | "answer" | "done" | "error", ...}

REST (all JSON): /api/meta, /api/scenarios, /api/scenarios/{id}, /api/scenarios/generate,
/api/models, /api/runs/{run}/memory/{mid}, /api/runs/{run}/trace/{mid}, /api/runs/{run}/graph
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from agents.callbacks import CallLogger
from agents.graph import stream_scenario
from agents.models import describe, enable_llm_cache, get_models
from data.automotive import SUPPORT, generate_plant
from data.simulator import validate
from eval.run_experiment import make_backend
from graphmem.config import PROJECT_ROOT, load_config
from graphmem.models import normalize_machine
from graphmem.retrieve import strength

SCENARIO_DIRS = {"plant": PROJECT_ROOT / "data/scenarios/plant", "dev": PROJECT_ROOT / "data/scenarios/dev"}
CONDITIONS = ["graph", "full", "window", "vector", "summary"]
MAX_RUNS = 8

app = FastAPI(title="GraphMem factory server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
CFG = load_config()
RUNS: "OrderedDict[str, Run]" = OrderedDict()


# ======================= scenarios =======================
def _scenario_files():
    for kind, d in SCENARIO_DIRS.items():
        for f in sorted(d.glob("*.json")) if d.exists() else []:
            yield kind, f


def load_scenario(sid: str) -> dict:
    for _, f in _scenario_files():
        if f.stem == sid:
            return json.loads(f.read_text(encoding="utf-8"))
    raise HTTPException(404, f"scenario {sid} not found")


def public_event(ev: dict) -> dict:
    """Event as shown in the UI; the answer key is only sent with the 'answer' message."""
    return {k: v for k, v in ev.items() if k not in ("gold", "needed_by", "duplicate_of", "corrects")}


@app.get("/api/meta")
def meta():
    return {"presets": list(CFG.agents.presets), "default_preset": CFG.agents.preset,
            "conditions": CONDITIONS, "modes": ["replay", "live"],
            "token_budget": CFG.retrieval.token_budget_default, "support_agents": SUPPORT}


@app.get("/api/scenarios")
def scenarios():
    out = []
    for kind, f in _scenario_files():
        s = json.loads(f.read_text(encoding="utf-8"))
        out.append({"id": s["scenario_id"], "kind": kind, "events": len(s["events"]),
                    "questions": sum(e["type"] == "question" for e in s["events"]),
                    "machines": s["machines"], "has_zones": bool(s.get("zones")),
                    "injected": bool(s.get("injected_error"))})
    return out


@app.post("/api/scenarios/generate")
def generate(n: int = 10, seed: int = 42):
    d = SCENARIO_DIRS["plant"]
    d.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        s = generate_plant(i, seed)
        validate(s)
        (d / f"{s['scenario_id']}.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    return {"generated": n}


@app.get("/api/scenarios/{sid}")
def scenario(sid: str):
    s = load_scenario(sid)
    return {**{k: v for k, v in s.items() if k != "events"},
            "events": [public_event(e) for e in s["events"]]}


@app.get("/api/models")
def models(preset: str | None = None):
    preset = preset or CFG.agents.preset
    chains = get_models(CFG, preset)
    return {"preset": preset, "text": describe(chains),
            "roles": {r: {"primary": c.primary, "fallbacks": [s.label for s in c.specs[1:]],
                          "skipped": c.skipped} for r, c in chains.items()}}


# ======================= runs =======================
class Run:
    def __init__(self, scenario: dict, opts: dict, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue):
        self.id = uuid.uuid4().hex[:8]
        self.scenario = scenario
        self.opts = opts
        self.loop, self.queue = loop, queue
        self.stop = threading.Event()
        self.status = "starting"
        cfg = CFG.copy()
        self.cfg = cfg
        self.preset = opts.get("preset") or cfg.agents.preset
        enable_llm_cache(cfg, self.preset)
        self.models = get_models(cfg, self.preset, seed=42)
        self.logger = CallLogger(condition=opts.get("condition", "graph"), scenario=scenario["scenario_id"], seed=42)
        self.memory = make_backend(opts.get("condition", "graph"), cfg, self.models, self.logger,
                                   scenario["scenario_id"])
        self.memory.listener = self.push
        self.logger.listener = self.push
        self.answers: list[dict] = []
        self.thread = threading.Thread(target=self.work, daemon=True)

    def push(self, kind: str, payload: dict):
        msg = {"type": kind, "run_id": self.id, "t": round(time.time(), 3), **payload}
        self.loop.call_soon_threadsafe(self.queue.put_nowait, msg)

    def work(self):
        self.status = "running"
        delay = float(self.opts.get("delay", 0.6))
        try:
            gen = stream_scenario(self.scenario, self.memory, self.models, self.logger, self.cfg,
                                  mode=self.opts.get("mode", "replay"), thread_id=self.id,
                                  token_budget=self.opts.get("budget"))
            for update in gen:
                if self.stop.is_set():
                    self.status = "stopped"
                    break
                for node, val in (update or {}).items():
                    if not isinstance(val, dict):
                        continue
                    if node == "next_event" and val.get("current"):
                        self.push("event", {"turn": val.get("turn"), "event": public_event(val["current"])})
                        if delay > 0:
                            time.sleep(delay)
                    if node == "report":
                        for a in val.get("answers", []):
                            gold, rep = a.get("gold") or {}, a.get("report") or {}
                            ok = rep.get("root_cause") == gold.get("root_cause") and \
                                normalize_machine(rep.get("machine")) == normalize_machine(gold.get("machine"))
                            a = {**a, "correct": ok}
                            self.answers.append(a)
                            self.push("answer", a)
            else:
                self.status = "done"
            n = len(self.answers)
            self.push("done", {"status": self.status, "answers": n,
                               "correct": sum(a["correct"] for a in self.answers),
                               "stats": self.memory.stats()})
        except Exception as e:
            self.status = "error"
            self.push("error", {"message": f"{type(e).__name__}: {e}"[:600]})


def get_run(rid: str) -> Run:
    if rid not in RUNS:
        raise HTTPException(404, "run not found (runs are kept in memory while the server is running)")
    return RUNS[rid]


def _retry(fn):
    for _ in range(3):                       # the run thread may be mutating the graph
        try:
            return fn()
        except RuntimeError:
            time.sleep(0.05)
    return fn()


@app.websocket("/ws/run")
async def ws_run(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    run: Run | None = None

    async def sender():
        while True:
            msg = await queue.get()
            await ws.send_text(json.dumps(msg, default=str))

    send_task = asyncio.create_task(sender())
    await ws.send_text(json.dumps({"type": "hello"}))
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            if msg.get("type") == "start":
                if run is not None:
                    run.stop.set()
                try:
                    sc = load_scenario(msg["scenario"])
                    run = Run(sc, msg, loop, queue)
                except Exception as e:
                    await queue.put({"type": "error", "message": f"{type(e).__name__}: {e}"[:600]})
                    continue
                RUNS[run.id] = run
                while len(RUNS) > MAX_RUNS:
                    RUNS.popitem(last=False)
                await queue.put({
                    "type": "run_started", "run_id": run.id, "scenario": scenario(sc["scenario_id"]),
                    "condition": msg.get("condition", "graph"), "preset": run.preset,
                    "mode": msg.get("mode", "replay"),
                    "models": {r: {"primary": c.primary, "skipped": c.skipped} for r, c in run.models.items()}})
                run.thread.start()
            elif msg.get("type") == "stop" and run is not None:
                run.stop.set()
    except WebSocketDisconnect:
        if run is not None:
            run.stop.set()
    finally:
        send_task.cancel()


@app.get("/api/runs/{rid}/memory/{mid}")
def memory_detail(rid: str, mid: str):
    run = get_run(rid)
    mem = run.memory
    if not hasattr(mem, "store"):                      # baseline: flat message list
        m = getattr(mem, "by_id", {}).get(mid)
        if not m:
            raise HTTPException(404, "memory not found")
        return {"id": mid, "text": m["text"], "kind": m["kind"], "agents": [m["agent"]], "turn": m["turn"],
                "originals": mem.expand(mid), "keywords": [], "derived_from": m["derived_from"]}

    def build():
        g = mem.store.g
        if mid not in g:
            raise HTTPException(404, "memory not found")
        d = g.nodes[mid]
        return {"id": mid, "text": d["text"], "kind": d["kind"], "turn": d["created_at"],
                "agents": mem.store.writers(mid), "readers": [v.split(":", 1)[1] for v in mem.store.out(mid, "read_by")],
                "sent_to": [v.split(":", 1)[1] for v in mem.store.out(mid, "sent_to")],
                "keywords": mem.keywords_text(mid), "derived_from": mem.store.out(mid, "derived_from"),
                "supersedes": mem.store.out(mid, "supersedes"), "superseded": d["valid_until"] is not None,
                "archived": d["archived"], "tokens": d["tokens"], "orig_tokens": d["orig_tokens"],
                "n_writers": d["n_writers"], "use_count": d["use_count"],
                "strength": round(strength(mem.store, mem.cfg, mid, mem.now), 3),
                "originals": mem.cold.get(mid)}
    return _retry(build)


@app.get("/api/runs/{rid}/trace/{mid}")
def trace(rid: str, mid: str):
    run = get_run(rid)
    try:
        return _retry(lambda: run.memory.trace(mid))
    except KeyError:
        raise HTTPException(404, "memory not found")


@app.get("/api/runs/{rid}/graph")
def graph(rid: str, limit: int = 300):
    """Active part of the memory graph: memories, keywords, agents and their links."""
    mem = get_run(rid).memory
    if not hasattr(mem, "store"):
        return {"nodes": [], "edges": []}

    def build():
        g = mem.store.g
        mems = [n for n, d in g.nodes(data=True) if d["type"] == "memory"][-limit:]
        keep = set(mems)
        nodes, edges = [], []
        for m in mems:
            d = g.nodes[m]
            nodes.append({"id": m, "type": "memory", "kind": d["kind"], "label": d["text"][:80],
                          "active": mem.store.is_active(m)})
            for _, v, k in g.out_edges(m, keys=True):
                if k in ("mentions", "written_by", "derived_from", "supersedes"):
                    edges.append({"source": m, "target": v, "type": k})
                    keep.add(v)
        for n in keep - set(mems):
            if n in g:
                d = g.nodes[n]
                nodes.append({"id": n, "type": d["type"],
                              "label": d.get("canonical") or d.get("name") or n})
        return {"nodes": nodes, "edges": edges}
    return _retry(build)


@app.get("/api/runs/{rid}")
def run_info(rid: str):
    run = get_run(rid)
    return {"id": run.id, "status": run.status, "scenario": run.scenario["scenario_id"],
            "answers": run.answers, "stats": run.memory.stats()}


# ======================= built UI =======================
UI_DIST = PROJECT_ROOT / "ui" / "dist"
if UI_DIST.exists():
    app.mount("/", StaticFiles(directory=str(UI_DIST), html=True), name="ui")
