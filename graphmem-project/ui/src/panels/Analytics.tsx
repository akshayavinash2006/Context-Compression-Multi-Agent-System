import { useStore, type TokenPoint } from "../store";

function Chart({ points }: { points: TokenPoint[] }) {
  const W = 320, H = 130, P = 6;
  if (points.length < 2) return <div className="chart-empty">The token chart fills in as agents call their LLMs.</div>;
  const maxY = Math.max(...points.map((p) => p.full), 1);
  const x = (i: number) => P + (i / (points.length - 1)) * (W - 2 * P);
  const y = (v: number) => H - P - (v / maxY) * (H - 2 * P);
  const line = (k: "graph" | "full") => points.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p[k]).toFixed(1)}`).join(" ");
  const area = `${line("graph")} L${x(points.length - 1)},${H - P} L${x(0)},${H - P} Z`;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label="Cumulative input tokens">
      <defs>
        <linearGradient id="g-area" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stopColor="#22d3ee" stopOpacity="0.35" /><stop offset="1" stopColor="#22d3ee" stopOpacity="0" />
        </linearGradient>
      </defs>
      {[0.25, 0.5, 0.75].map((f) => <line key={f} x1={P} x2={W - P} y1={y(maxY * f)} y2={y(maxY * f)} className="gridline" />)}
      <path d={line("full")} className="line-full" />
      <path d={area} fill="url(#g-area)" />
      <path d={line("graph")} className="line-graph" />
    </svg>
  );
}

export function Analytics() {
  const { tokens, graphCum, fullCum, counts, condition, answers } = useStore();
  const saved = fullCum > 0 ? Math.round((1 - graphCum / fullCum) * 100) : 0;
  const correct = answers.filter((a) => a.correct).length;
  const ratio = counts.storedTokens > 0 && condition === "graph" ? (counts.origTokens / Math.max(1, counts.storedTokens)) : null;
  return (
    <section className="card analytics">
      <div className="card-head"><h2>Memory Analytics</h2><span className="tag">{condition === "graph" ? "Graph memory" : condition}</span></div>
      <div className="kpi">
        <div className="kpi-big">{graphCum.toLocaleString()}<small> tokens sent</small></div>
        {condition !== "full" && fullCum > 0 && <div className="kpi-delta">{saved}% fewer than full history <small>(est. {fullCum.toLocaleString()})</small></div>}
      </div>
      <Chart points={tokens} />
      <div className="legend"><span><i className="sw graph" />This run</span><span><i className="sw full" />Full history (estimated)</span></div>
      <div className="stat-grid">
        <div><b>{counts.active}</b><span>Active memories</span></div>
        <div><b>{counts.merge}</b><span>Duplicates merged</span></div>
        <div><b>{counts.update}</b><span>Facts replaced</span></div>
        <div><b>{ratio ? ratio.toFixed(2) + "×" : "–"}</b><span>Compression</span></div>
        <div><b>{answers.length ? `${correct}/${answers.length}` : "–"}</b><span>Correct answers</span></div>
        <div><b>{tokens.length}</b><span>LLM calls</span></div>
      </div>
    </section>
  );
}
