import { useEffect, useRef, useState } from "react";
import { Empty, Page } from "../components/ui";
import { getJSON } from "../api";
import { ZONES, zoneFor } from "../layout";
import { useStore } from "../store";

interface GNode { id: string; type: string; kind?: string; label: string; active?: boolean }
interface GEdge { source: string; target: string; type: string }
interface Sim extends GNode { x: number; y: number; vx: number; vy: number }

const W = 1000, H = 620;
const COLOR: Record<string, string> = { memory: "#2ee6d0", keyword: "#fbbf24", agent: "#a78bfa", entity: "#7dd3fc" };
const KIND_COLOR: Record<string, string> = { observation: "#38bdf8", diagnosis: "#f472b6", report: "#c4b5fd", summary: "#4ade80" };
const nodeColor = (n: GNode) => (n.type === "memory" ? KIND_COLOR[n.kind ?? ""] ?? COLOR.memory : COLOR[n.type] ?? "#94a3b8");

export function MemoryGraph() {
  const { runId, wsStatus, selectedAgent, set } = useStore();
  const [nodes, setNodes] = useState<Sim[]>([]);
  const [edges, setEdges] = useState<GEdge[]>([]);
  const [hover, setHover] = useState<Sim>();
  const sim = useRef<Map<string, Sim>>(new Map());
  const edgeRef = useRef<GEdge[]>([]);
  const alpha = useRef(1);

  // Pull the graph now and every few seconds while a run is live; keep old positions so it grows smoothly.
  useEffect(() => {
    if (!runId) return;
    let dead = false;
    const pull = () => getJSON<{ nodes: GNode[]; edges: GEdge[] }>(`/api/runs/${runId}/graph?limit=160`).then((g) => {
      if (dead) return;
      const next = new Map<string, Sim>();
      for (const n of g.nodes) {
        const old = sim.current.get(n.id);
        next.set(n.id, old ? { ...old, ...n } : { ...n, x: W / 2 + (Math.random() - 0.5) * 200, y: H / 2 + (Math.random() - 0.5) * 200, vx: 0, vy: 0 });
      }
      sim.current = next;
      edgeRef.current = g.edges.filter((e) => next.has(e.source) && next.has(e.target));
      setEdges(edgeRef.current);
      alpha.current = 1;
    }).catch(() => {});
    pull();
    const id = window.setInterval(pull, wsStatus === "running" ? 2500 : 60000);
    return () => { dead = true; window.clearInterval(id); };
  }, [runId, wsStatus]);

  useEffect(() => {
    let raf = 0;
    const step = () => {
      const ns = [...sim.current.values()];
      if (ns.length && alpha.current > 0.02) {
        const a = alpha.current;
        for (let i = 0; i < ns.length; i++) {
          const p = ns[i];
          for (let j = i + 1; j < ns.length; j++) {
            const q = ns[j];
            let dx = p.x - q.x, dy = p.y - q.y;
            const d2 = dx * dx + dy * dy + 0.5;
            if (d2 > 90000) continue;
            const f = (1800 / d2) * a;
            const d = Math.sqrt(d2);
            dx /= d; dy /= d;
            p.vx += dx * f; p.vy += dy * f; q.vx -= dx * f; q.vy -= dy * f;
          }
          p.vx += (W / 2 - p.x) * 0.004 * a; p.vy += (H / 2 - p.y) * 0.004 * a;
        }
        for (const e of edgeRef.current) {
          const s = sim.current.get(e.source), t = sim.current.get(e.target);
          if (!s || !t) continue;
          const dx = t.x - s.x, dy = t.y - s.y, d = Math.sqrt(dx * dx + dy * dy) || 1;
          const f = (d - 70) * 0.02 * a;
          s.vx += (dx / d) * f; s.vy += (dy / d) * f; t.vx -= (dx / d) * f; t.vy -= (dy / d) * f;
        }
        for (const n of ns) {
          n.vx *= 0.82; n.vy *= 0.82;
          n.x = Math.max(24, Math.min(W - 24, n.x + n.vx)); n.y = Math.max(24, Math.min(H - 24, n.y + n.vy));
        }
        alpha.current *= 0.985;
        setNodes([...ns]);
      }
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, []);

  const byId = new Map(nodes.map((n) => [n.id, n]));
  // Agent focus: light up what the selected agent committed (written_by edges) plus their keywords.
  const focusId = selectedAgent ? `A:${selectedAgent}` : undefined;
  const hot = new Set<string>();
  if (focusId && byId.has(focusId)) {
    hot.add(focusId);
    for (const e of edges) if (e.type === "written_by" && e.target === focusId) hot.add(e.source);
    for (const e of edges) if (e.type === "mentions" && hot.has(e.source)) hot.add(e.target);
  }
  const focusing = hot.size > 0;
  const committed = focusing ? [...hot].filter((id) => byId.get(id)?.type === "memory").length : 0;
  const focusColor = selectedAgent ? zoneFor(selectedAgent).color : undefined;
  const edgeHot = (e: GEdge) => hot.has(e.source) && hot.has(e.target);
  return (
    <Page title="Memory Graph" sub="Memories, the keywords they mention and the agents that wrote them. Pick an agent to highlight what it committed; click a memory to inspect it.">
      <div className="focus-bar">
        <span className="muted">Focus</span>
        <button className={`focus-chip ${!selectedAgent ? "on" : ""}`} onClick={() => set({ selectedAgent: undefined })}>Everything</button>
        {ZONES.map((z) => (
          <button key={z.id} className={`focus-chip ${selectedAgent === z.id ? "on" : ""}`} style={{ ["--c" as string]: z.color }}
            onClick={() => set({ selectedAgent: selectedAgent === z.id ? undefined : z.id })}><i />{z.name}</button>
        ))}
        {selectedAgent && <span className="focus-note">{focusing ? `${committed} memories committed by ${zoneFor(selectedAgent).name}` : `${zoneFor(selectedAgent).name} has not committed anything in view yet`}</span>}
      </div>
      <div className="graph-card card">
        {nodes.length === 0 ? <Empty>{runId ? "Loading the graph…" : "Start a run and the graph builds here as agents write."}</Empty> : (
          <>
            <svg viewBox={`0 0 ${W} ${H}`} className="graph" role="img" aria-label="Memory graph">
              {edges.map((e, i) => {
                const s = byId.get(e.source), t = byId.get(e.target);
                return s && t ? <line key={i} x1={s.x} y1={s.y} x2={t.x} y2={t.y} className={`g-edge ${e.type} ${focusing ? (edgeHot(e) ? "hot" : "dim") : ""}`}
                  style={focusing && edgeHot(e) && e.type === "written_by" ? { stroke: focusColor } : undefined} /> : null;
              })}
              {nodes.map((n) => (
                <g key={n.id} transform={`translate(${n.x},${n.y})`} className={`g-node ${n.type} ${n.active === false ? "off" : ""} ${focusing ? (hot.has(n.id) ? "hot" : "dim") : ""}`}
                  onMouseEnter={() => setHover(n)} onMouseLeave={() => setHover(undefined)}
                  onClick={() => n.type === "memory" && set({ openMemory: n.id })}>
                  {focusing && hot.has(n.id) && n.type === "memory" && <circle r={11} className="g-ring" style={{ stroke: focusColor }} />}
                  <circle r={n.type === "memory" ? 7 : n.type === "agent" ? (n.id === focusId ? 12 : 9) : 4.5} fill={n.id === focusId ? focusColor : nodeColor(n)} style={{ filter: `drop-shadow(0 0 6px ${n.id === focusId ? focusColor : nodeColor(n)})` }} />
                </g>
              ))}
            </svg>
            <div className="graph-key">
              <span><i style={{ background: KIND_COLOR.observation }} />Observation</span>
              <span><i style={{ background: KIND_COLOR.diagnosis }} />Diagnosis</span>
              <span><i style={{ background: KIND_COLOR.report }} />Report</span>
              <span><i style={{ background: COLOR.keyword }} />Keyword</span>
              <span><i style={{ background: COLOR.agent }} />Agent</span>
            </div>
            {hover && <div className="graph-tip"><b>{hover.type === "memory" ? hover.id : hover.type}</b><p>{hover.label}</p></div>}
          </>
        )}
      </div>
    </Page>
  );
}
