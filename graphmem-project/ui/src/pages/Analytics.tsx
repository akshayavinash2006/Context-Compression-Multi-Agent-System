import { motion } from "framer-motion";
import { Card, Donut, Empty, Gauge, Page, Stat, stagger } from "../components/ui";
import { ZONE_BY_ID } from "../layout";
import { useStore, type TokenPoint } from "../store";

function Chart({ points }: { points: TokenPoint[] }) {
  const W = 720, H = 260, PX = 8, PT = 14, PB = 22;
  if (points.length < 2) return <Empty>The chart fills in as agents call their LLMs.</Empty>;
  const maxY = Math.max(...points.map((p) => p.full), 1);
  const x = (i: number) => PX + (i / (points.length - 1)) * (W - 2 * PX);
  const y = (v: number) => H - PB - (v / maxY) * (H - PB - PT);
  const line = (k: "graph" | "full") => points.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p[k]).toFixed(1)}`).join(" ");
  const gl = line("graph");
  const last = points[points.length - 1];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label="Cumulative input tokens, this run against full history">
      <defs>
        <linearGradient id="ga" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="#2ee6d0" stopOpacity=".38" /><stop offset="1" stopColor="#2ee6d0" stopOpacity="0" /></linearGradient>
      </defs>
      {[0, 0.25, 0.5, 0.75, 1].map((f) => (
        <g key={f}>
          <line x1={PX} x2={W - PX} y1={y(maxY * f)} y2={y(maxY * f)} className="gridline" />
          {f > 0 && <text x={W - PX} y={y(maxY * f) - 4} className="axis" textAnchor="end">{Math.round(maxY * f).toLocaleString()}</text>}
        </g>
      ))}
      <path d={line("full")} className="line-full" />
      <path d={`${gl} L${x(points.length - 1)},${H - PB} L${x(0)},${H - PB} Z`} fill="url(#ga)" />
      <path d={gl} className="line-graph" />
      <circle cx={x(points.length - 1)} cy={y(last.graph)} r="4.5" className="end-dot" />
      <text x={PX} y={H - 4} className="axis">call 1</text>
      <text x={W - PX} y={H - 4} className="axis" textAnchor="end">call {points.length}</text>
    </svg>
  );
}

export function Analytics() {
  const { tokens, graphCum, fullCum, counts, condition, answers, agents } = useStore();
  const saved = fullCum > 0 ? Math.round((1 - graphCum / fullCum) * 100) : 0;
  const correct = answers.filter((a) => a.correct).length;
  const ratio = counts.storedTokens > 0 && condition === "graph" ? counts.origTokens / Math.max(1, counts.storedTokens) : 0;
  const perAgent = Object.entries(agents).filter(([, a]) => a.inputTokens > 0).sort((a, b) => b[1].inputTokens - a[1].inputTokens);
  const maxTok = Math.max(1, ...perAgent.map(([, a]) => a.inputTokens));
  const decisions = [
    { label: "Created", value: counts.create, color: "#38bdf8" },
    { label: "Merged", value: counts.merge, color: "#4ade80" },
    { label: "Replaced", value: counts.update, color: "#f87171" },
    { label: "Appended", value: counts.append, color: "#94a3b8" },
  ];
  const writes = decisions.reduce((a, d) => a + d.value, 0);

  return (
    <Page title="Memory Analytics" sub="How much the shared memory saves, and how healthy it stays.">
      <motion.div className="stat-row" variants={stagger}>
        <Stat label="Tokens sent" value={graphCum} />
        <Stat label="Saved vs full history" value={saved} suffix="%" tone="var(--ok)" note={fullCum ? `est. ${fullCum.toLocaleString()} without memory` : undefined} />
        <Stat label="Active memories" value={counts.active} />
        <Stat label="Compression" value={ratio ? ratio : "–"} decimals={2} suffix={ratio ? "×" : ""} />
      </motion.div>

      <div className="grid-main">
        <Card title="Cumulative input tokens" aside={<div className="legend"><span><i className="sw g" />This run</span><span><i className="sw f" />Full history (est.)</span></div>}>
          <Chart points={tokens} />
        </Card>
        <Card title="Answer accuracy">
          <div className="center-col">
            <Gauge value={answers.length ? correct / answers.length : 0} label={answers.length ? `${correct} of ${answers.length} correct` : "No answers yet"} color="var(--ok)" />
          </div>
        </Card>
      </div>

      <div className="grid-3">
        <Card title="Write decisions">
          <div className="donut-row">
            <Donut parts={decisions} center={writes} sub="writes" />
            <ul className="key">{decisions.map((d) => <li key={d.label}><i style={{ background: d.color }} />{d.label}<b>{d.value}</b></li>)}</ul>
          </div>
        </Card>
        <Card title="Input tokens by agent">
          {perAgent.length === 0 ? <Empty>Agents that call a model show up here.</Empty> : (
            <ul className="bars">
              {perAgent.map(([id, a]) => (
                <li key={id}>
                  <span>{ZONE_BY_ID[id]?.name ?? id}</span>
                  <div className="bar"><motion.i initial={{ width: 0 }} animate={{ width: `${(a.inputTokens / maxTok) * 100}%` }} transition={{ duration: 0.7, ease: "easeOut" }} style={{ background: ZONE_BY_ID[id]?.color }} /></div>
                  <b>{a.inputTokens.toLocaleString()}</b>
                </li>
              ))}
            </ul>
          )}
        </Card>
        <Card title="Memory health">
          <div className="mini-grid">
            <div><b>{counts.merge}</b><span>Duplicates merged</span></div>
            <div><b>{counts.update}</b><span>Facts replaced</span></div>
            <div><b>{counts.superseded}</b><span>Superseded</span></div>
            <div><b>{tokens.length}</b><span>LLM calls</span></div>
          </div>
        </Card>
      </div>
    </Page>
  );
}
