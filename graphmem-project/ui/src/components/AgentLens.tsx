import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { zoneFor } from "../layout";
import { go } from "../router";
import { STATUS_COLOR, STATUS_LABEL } from "../status";
import { decisionColor, useStore, type ContextFlow, type MemInfo } from "../store";

// Left to right: agents it heard from -> facts it was given -> the agent -> facts it committed -> agents those reached.
const W = 1100;
const COL = { src: 92, recv: 330, me: 550, out: 770, dst: 1008 };
const CARD_W = 184, CARD_H = 38, AGENT_W = 132, ME_W = 150;
const MAX_MEM = 6, MAX_AGENTS = 5;
const RECV_COLOR = "#fbbf24";

const clip = (s: string, n: number) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);
const curve = (x1: number, y1: number, x2: number, y2: number) => {
  const dx = (x2 - x1) * 0.5;
  return `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`;
};
const rows = (n: number, h: number) => Array.from({ length: n }, (_, i) => (h / (n + 1)) * (i + 1));

interface Tip { x: number; y: number; title: string; color: string; lines: string[] }

export function AgentLens({ id }: { id: string }) {
  const { commits, memories, lastRead, contextFlows, agents, set } = useStore();
  const [hover, setHover] = useState<string>();     // node key: "m:<id>" or "a:<agent>"
  const [tip, setTip] = useState<Tip>();
  const zone = zoneFor(id);
  const status = agents[id]?.status ?? "idle";

  const g = useMemo(() => {
    const mine = commits[id] ?? [];
    // Latest commit per memory, newest first.
    const seen = new Set<string>();
    const out: { mem: MemInfo; decision: string; turn: number; times: number }[] = [];
    for (let i = mine.length - 1; i >= 0; i--) {
      const c = mine[i];
      if (seen.has(c.memoryId)) continue;
      seen.add(c.memoryId);
      const mem = memories[c.memoryId];
      if (mem) out.push({ mem, decision: c.decision, turn: c.turn, times: mine.filter((x) => x.memoryId === c.memoryId).length });
    }
    const handoff = contextFlows.filter((f) => f.to === id && f.via.includes("handoff")).flatMap((f) => f.memories.slice(-1));
    const recvIds = [...new Set([...handoff, ...(lastRead[id]?.memory_ids ?? [])])];
    const recv = recvIds.map((m) => memories[m]).filter(Boolean) as MemInfo[];
    const byCount = (a: ContextFlow, b: ContextFlow) => b.count - a.count;
    const srcFlows = contextFlows.filter((f) => f.to === id).sort(byCount);
    const dstFlows = contextFlows.filter((f) => f.from === id).sort(byCount);
    return {
      out: out.slice(0, MAX_MEM), outMore: Math.max(0, out.length - MAX_MEM), totalCommits: mine.length,
      recv: recv.slice(0, MAX_MEM), recvMore: Math.max(0, recv.length - MAX_MEM), handoff: new Set(handoff),
      src: srcFlows.slice(0, MAX_AGENTS), dst: dstFlows.slice(0, MAX_AGENTS),
      srcAll: srcFlows.length, dstAll: dstFlows.length,
    };
  }, [id, commits, memories, lastRead, contextFlows]);

  const n = Math.max(g.out.length, g.recv.length, g.src.length, g.dst.length, 3);
  const H = Math.max(250, n * 50 + 40);
  const ys = { out: rows(g.out.length, H), recv: rows(g.recv.length, H), src: rows(g.src.length, H), dst: rows(g.dst.length, H) };
  const midY = H / 2;
  const posOf = { src: new Map(g.src.map((f, i) => [f.from, ys.src[i]])), dst: new Map(g.dst.map((f, i) => [f.to, ys.dst[i]])) };

  // Edges, each tagged with the node keys it touches so hover can light up a path.
  type Edge = { d: string; color: string; cls: string; keys: string[]; w: number };
  const edges: Edge[] = [];
  g.recv.forEach((m, i) => {
    const y = ys.recv[i];
    edges.push({ d: curve(COL.recv + CARD_W / 2, y, COL.me - ME_W / 2, midY), color: g.handoff.has(m.id) ? "#f472b6" : RECV_COLOR,
      cls: g.handoff.has(m.id) ? "handoff" : "recv", keys: [`m:${m.id}`], w: 1.6 });
    for (const a of m.authors) {
      const sy = posOf.src.get(a);
      if (sy !== undefined) edges.push({ d: curve(COL.src + AGENT_W / 2, sy, COL.recv - CARD_W / 2, y), color: zoneFor(a).color, cls: "src", keys: [`a:${a}`, `m:${m.id}`], w: 1.2 });
    }
  });
  // Sources whose facts were pushed to this agent but are not in its last read: straight to the agent's inbox.
  const linked = new Set(g.recv.flatMap((m) => m.authors));
  g.src.forEach((f, i) => {
    if (!linked.has(f.from)) edges.push({ d: curve(COL.src + AGENT_W / 2, ys.src[i], COL.me - ME_W / 2, midY), color: zoneFor(f.from).color, cls: "inbox", keys: [`a:${f.from}`], w: 1 });
  });
  g.out.forEach((o, i) => {
    const y = ys.out[i];
    edges.push({ d: curve(COL.me + ME_W / 2, midY, COL.out - CARD_W / 2, y), color: zone.color, cls: i === 0 ? "commit live" : "commit", keys: [`m:${o.mem.id}`], w: i === 0 ? 2.6 : 1.8 });
    for (const a of new Set([...o.mem.readers, ...o.mem.pushedTo])) {
      const dy = posOf.dst.get(a);
      if (dy !== undefined && a !== id) edges.push({ d: curve(COL.out + CARD_W / 2, y, COL.dst - AGENT_W / 2, dy), color: zoneFor(a).color, cls: "dst", keys: [`m:${o.mem.id}`, `a:${a}`], w: 1.2 });
    }
  });
  const lit = (keys: string[]) => !hover || keys.includes(hover);

  const memTip = (m: MemInfo, x: number, y: number, head: string, color: string) => setTip({ x, y, color, title: `${m.id} · ${head}`, lines: [
    m.text,
    `Written by ${m.authors.map((a) => zoneFor(a).name).join(", ")}`,
    m.readers.length ? `Read by ${m.readers.map((a) => zoneFor(a).name).join(", ")}` : "Not read by anyone yet",
    ...(m.supersededBy ? [`Replaced by ${m.supersededBy}`] : []),
    ...(m.origTokens ? [`${m.tokens} tokens stored (from ${m.origTokens})`] : []),
  ] });
  const enter = (key: string) => setHover(key);
  const leave = () => { setHover(undefined); setTip(undefined); };

  const empty = g.totalCommits === 0 && g.recv.length === 0;
  return (
    <motion.section className="lens glass" style={{ ["--accent" as string]: zone.color }}
      initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 24, transition: { duration: 0.15 } }}
      transition={{ type: "spring", stiffness: 280, damping: 30 }}>
      <header className="lens-head">
        <div className="lens-title">
          <span className="eyebrow">Agent lens</span>
          <h2>{zone.name}</h2>
          <span className="pill" style={{ ["--c" as string]: STATUS_COLOR[status] }}><i />{STATUS_LABEL[status]}</span>
        </div>
        <div className="lens-stats">
          <span><b>{g.totalCommits}</b> commits</span>
          <span><b>{lastRead[id]?.memory_ids.length ?? 0}</b> facts in last read</span>
          <span><b>{g.srcAll}</b> sources</span>
          <span><b>{g.dstAll}</b> listeners</span>
        </div>
        <div className="lens-actions">
          <button className="link-btn" onClick={() => go("graph")}>Open in memory graph →</button>
          <button className="icon-btn" onClick={() => set({ selectedAgent: undefined })} aria-label="Close agent lens">×</button>
        </div>
      </header>

      {empty ? (
        <div className="lens-empty">Nothing yet. When {zone.name} reads or commits a fact, it shows up here.</div>
      ) : (
        <div className="lens-canvas" onMouseLeave={leave}>
          <div className="lens-cols" aria-hidden>
            <span style={{ left: `${(COL.src / W) * 100}%` }}>Heard from</span>
            <span style={{ left: `${(COL.recv / W) * 100}%` }}>Received</span>
            <span style={{ left: `${(COL.out / W) * 100}%`, color: zone.color }}>Committed</span>
            <span style={{ left: `${(COL.dst / W) * 100}%` }}>Reached</span>
          </div>
          <svg viewBox={`0 0 ${W} ${H}`} className="lens-svg" role="img" aria-label={`What ${zone.name} received and committed`}>
            <g>
              {edges.map((e, i) => (
                <path key={i} d={e.d} className={`l-edge ${e.cls} ${lit(e.keys) ? "" : "dim"} ${hover && lit(e.keys) ? "hot" : ""}`}
                  stroke={e.color} strokeWidth={e.w} />
              ))}
            </g>

            {/* sources */}
            {g.src.map((f, i) => (
              <AgentNode key={f.from} x={COL.src} y={ys.src[i]} agent={f.from} sub={`${f.count} fact${f.count > 1 ? "s" : ""}${f.via.includes("handoff") ? " · hand-off" : ""}`}
                dim={!!hover && hover !== `a:${f.from}`} onEnter={() => enter(`a:${f.from}`)} onLeave={leave} onClick={() => set({ selectedAgent: f.from })} />
            ))}

            {/* received memories */}
            <AnimatePresence>
              {g.recv.map((m, i) => {
                const own = m.authors.includes(id);
                const hand = g.handoff.has(m.id);
                const color = hand ? "#f472b6" : RECV_COLOR;
                return (
                  <MemNode key={`r-${m.id}`} x={COL.recv} y={ys.recv[i]} mem={m} color={color} tag={hand ? "HAND-OFF" : own ? "OWN" : "READ"}
                    dim={!!hover && hover !== `m:${m.id}` && !m.authors.some((a) => hover === `a:${a}`)}
                    onEnter={() => { enter(`m:${m.id}`); memTip(m, COL.recv, ys.recv[i], hand ? "handed off by Control Room" : "delivered in last read", color); }}
                    onLeave={leave} onClick={() => set({ openMemory: m.id })} />
                );
              })}
            </AnimatePresence>
            {g.recvMore > 0 && <text x={COL.recv} y={H - 6} className="l-more" textAnchor="middle">+{g.recvMore} more</text>}
            {g.recv.length === 0 && <ColumnNote x={COL.recv} y={midY} lines={["Has not read from memory yet", "replay mode: only Control Room", "and Reports read"]} />}
            {g.out.length === 0 && <ColumnNote x={COL.out} y={midY} lines={["No commits yet"]} />}

            {/* the agent */}
            <g transform={`translate(${COL.me},${midY})`} className="l-me">
              <rect x={-ME_W / 2} y={-34} width={ME_W} height={68} rx={16} stroke={zone.color} style={{ filter: `drop-shadow(0 0 14px ${zone.color}66)` }} />
              <circle cx={-ME_W / 2 + 18} cy={-12} r={4.5} fill={STATUS_COLOR[status]} className={status !== "idle" ? "l-pulse" : ""} />
              <text x={-ME_W / 2 + 30} y={-8} className="l-me-name">{clip(zone.name, 14)}</text>
              <text x={-ME_W / 2 + 18} y={16} className="l-me-sub">{zone.kind === "support" ? "support agent" : "zone agent"}</text>
            </g>

            {/* committed memories */}
            <AnimatePresence>
              {g.out.map((o, i) => {
                const replaced = !!o.mem.supersededBy;
                return (
                  <MemNode key={`o-${o.mem.id}`} x={COL.out} y={ys.out[i]} mem={o.mem} color={zone.color} committed latest={i === 0} replaced={replaced}
                    tag={o.decision.toUpperCase()} tagColor={decisionColor(o.decision)} times={o.times}
                    dim={!!hover && hover !== `m:${o.mem.id}` && ![...o.mem.readers, ...o.mem.pushedTo].some((a) => hover === `a:${a}`)}
                    onEnter={() => { enter(`m:${o.mem.id}`); memTip(o.mem, COL.out, ys.out[i], `${o.decision} at turn ${o.turn}`, zone.color); }}
                    onLeave={leave} onClick={() => set({ openMemory: o.mem.id })} />
                );
              })}
            </AnimatePresence>
            {g.outMore > 0 && <text x={COL.out} y={H - 6} className="l-more" textAnchor="middle">+{g.outMore} earlier</text>}

            {/* listeners */}
            {g.dst.map((f, i) => (
              <AgentNode key={f.to} x={COL.dst} y={ys.dst[i]} agent={f.to} sub={`${f.count} × ${f.via.join(" / ")}`}
                dim={!!hover && hover !== `a:${f.to}`} onEnter={() => enter(`a:${f.to}`)} onLeave={leave} onClick={() => set({ selectedAgent: f.to })} />
            ))}
          </svg>

          <AnimatePresence>
            {tip && (
              <motion.div className="lens-tip" style={{ left: `${(tip.x / W) * 100}%`, top: `${(tip.y / H) * 100}%`, ["--c" as string]: tip.color }}
                initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.12 }}>
                <b>{tip.title}</b>
                <p className="lt-text">{tip.lines[0]}</p>
                {tip.lines.slice(1).map((l) => <p key={l}>{l}</p>)}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}

      <footer className="lens-key">
        <span><i style={{ background: zone.color }} />Committed by {zone.name}</span>
        <span><i style={{ background: RECV_COLOR }} />Delivered from memory</span>
        <span><i style={{ background: "#f472b6" }} />Hand-off</span>
        <span><i className="dotted" />Pushed to inbox</span>
        <span className="lk-tags">
          <em style={{ color: decisionColor("create") }}>CREATE</em> new fact
          <em style={{ color: decisionColor("merge") }}>MERGE</em> duplicate
          <em style={{ color: decisionColor("update") }}>UPDATE</em> replaces old
        </span>
        <span className="lk-hint">Hover to trace a path · click a fact for its sources</span>
      </footer>
    </motion.section>
  );
}

function ColumnNote({ x, y, lines }: { x: number; y: number; lines: string[] }) {
  return (
    <text x={x} y={y - (lines.length - 1) * 8} textAnchor="middle" className="l-note">
      {lines.map((l, i) => <tspan key={l} x={x} dy={i ? 17 : 0} className={i ? "sub" : ""}>{l}</tspan>)}
    </text>
  );
}

function AgentNode({ x, y, agent, sub, dim, onEnter, onLeave, onClick }: { x: number; y: number; agent: string; sub: string; dim: boolean; onEnter: () => void; onLeave: () => void; onClick: () => void }) {
  const z = zoneFor(agent);
  return (
    <motion.g className={`l-agent ${dim ? "dim" : ""}`} initial={{ opacity: 0, x, y }} animate={{ opacity: 1, x, y }}
      onMouseEnter={onEnter} onMouseLeave={onLeave} onClick={onClick}>
      <rect x={-AGENT_W / 2} y={-19} width={AGENT_W} height={38} rx={19} stroke={z.color} />
      <circle cx={-AGENT_W / 2 + 15} cy={0} r={4} fill={z.color} />
      <text x={-AGENT_W / 2 + 26} y={-2} className="l-agent-name">{clip(z.name, 14)}</text>
      <text x={-AGENT_W / 2 + 26} y={12} className="l-agent-sub">{clip(sub, 19)}</text>
    </motion.g>
  );
}

function MemNode({ x, y, mem, color, tag, tagColor, committed, latest, replaced, times, dim, onEnter, onLeave, onClick }: {
  x: number; y: number; mem: MemInfo; color: string; tag: string; tagColor?: string; committed?: boolean; latest?: boolean;
  replaced?: boolean; times?: number; dim: boolean; onEnter: () => void; onLeave: () => void; onClick: () => void;
}) {
  return (
    <motion.g className={`l-mem ${committed ? "committed" : ""} ${latest ? "latest" : ""} ${replaced ? "replaced" : ""} ${dim ? "dim" : ""}`}
      initial={{ opacity: 0, x: x + (committed ? -24 : 24), y }} animate={{ opacity: 1, x, y }} exit={{ opacity: 0 }}
      transition={{ type: "spring", stiffness: 260, damping: 28 }}
      onMouseEnter={onEnter} onMouseLeave={onLeave} onClick={onClick} style={{ ["--c" as string]: color }}>
      <rect x={-CARD_W / 2} y={-CARD_H / 2} width={CARD_W} height={CARD_H} rx={10} />
      <rect x={-CARD_W / 2} y={-CARD_H / 2} width={3} height={CARD_H} rx={1.5} fill={color} className="l-bar" />
      <text x={-CARD_W / 2 + 12} y={-3} className="l-id">{mem.id}</text>
      <text x={CARD_W / 2 - 10} y={-3} className="l-tag" textAnchor="end" fill={tagColor ?? color}>{tag}{times && times > 1 ? ` ×${times}` : ""}</text>
      <text x={-CARD_W / 2 + 12} y={12} className="l-text">{clip(mem.text, 30)}</text>
    </motion.g>
  );
}
