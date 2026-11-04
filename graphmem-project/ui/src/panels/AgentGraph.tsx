import { useStore } from "../store";
import { zoneFor } from "../layout";

const POS: Record<string, [number, number]> = {
  sensor: [110, 105], history: [110, 235], diagnosis: [380, 170], report: [650, 170],
};
const ORDER = ["sensor", "history", "diagnosis", "report"];
const LABEL: Record<string, string> = { sensor: "Sensor", history: "History", diagnosis: "Diagnosis", report: "Report" };
const COLOR: Record<string, string> = { sensor: "#22d3ee", history: "#fbbf24", diagnosis: "#a78bfa", report: "#4ade80" };

export function AgentGraph() {
  const { contextFlows, agents, models } = useStore();
  const setStore = useStore((s) => s.set);
  const active = useStore((s) => s.agentGraphOpen);
  if (!active) return null;
  return (
    <div className="modal-backdrop" onClick={() => setStore({ agentGraphOpen: false })}>
      <section className="modal agent-graph-modal" aria-label="Agent context flow" onClick={(e) => e.stopPropagation()}>
        <div className="card-head"><div><h2>Agent context flow</h2><p className="muted">Live memory routing between agents</p></div>
          <button className="ghost small" onClick={() => setStore({ agentGraphOpen: false })}>Close</button></div>
        <div className="agent-graph-canvas">
          <svg viewBox="0 0 760 340" role="img" aria-label="Directed graph of agent context flows">
            <defs><marker id="flow-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#64748b" /></marker></defs>
            <path className="topology-edge" d="M195 105 C270 105 265 145 300 160" markerEnd="url(#flow-arrow)" />
            <path className="topology-edge" d="M195 235 C270 235 265 195 300 180" markerEnd="url(#flow-arrow)" />
            <path className="topology-edge" d="M465 170 L565 170" markerEnd="url(#flow-arrow)" />
            {contextFlows.map((flow) => {
              const a = POS[flow.from], b = POS[flow.to];
              if (!a || !b) return null;
              const color = COLOR[flow.from] ?? zoneFor(flow.from).color;
              const bend = a[1] === b[1] ? 0 : (b[1] > a[1] ? 25 : -25);
              return <path key={`${flow.from}-${flow.to}`} d={`M${a[0] + 80} ${a[1]} Q${(a[0] + b[0]) / 2} ${(a[1] + b[1]) / 2 + bend} ${b[0] - 85} ${b[1]}`} stroke={color} markerEnd="url(#flow-arrow)" className="context-edge" />;
            })}
            {ORDER.map((name) => {
              const [x, y] = POS[name], color = COLOR[name], agent = agents[name];
              return <g key={name} transform={`translate(${x},${y})`}>
                <rect x="-82" y="-34" width="164" height="68" rx="14" stroke={color} className={`agent-node ${agent?.status !== "idle" ? "agent-node-active" : ""}`} />
                <text textAnchor="middle" y="-4" className="agent-node-title">{LABEL[name]}</text>
                <text textAnchor="middle" y="17" className="agent-node-model">{models[name]?.primary ?? "model not loaded"}</text>
              </g>;
            })}
          </svg>
        </div>
        <div className="flow-legend"><span><i className="topology-swatch" />Agent workflow</span><span><i className="context-swatch" />Observed context transfer</span></div>
        <div className="flow-list">
          {!contextFlows.length && <p className="empty">Context edges will appear as agents share and read memories during a run.</p>}
          {[...contextFlows].reverse().map((flow) => <div className="flow-row" key={`${flow.from}-${flow.to}`}>
            <b style={{ color: COLOR[flow.from] }}>{LABEL[flow.from] ?? flow.from}</b><span>→</span>
            <b style={{ color: COLOR[flow.to] }}>{LABEL[flow.to] ?? flow.to}</b>
            <span className="flow-detail">{flow.memories.join(", ")} · {flow.tokens || "push"} tokens · turn {flow.lastTurn}</span>
          </div>)}
        </div>
      </section>
    </div>
  );
}
