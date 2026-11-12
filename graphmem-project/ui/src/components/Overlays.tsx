import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { getJSON } from "../api";
import { zoneFor } from "../layout";
import { useStore } from "../store";
import { parseReport, readable } from "../report";
import { ReportCard } from "./ReportCard";

interface Original { agent: string; turn: number; text: string; event_id?: string }
interface MemoryDetail { id: string; text: string; kind: string; turn: number; agents: string[]; readers?: string[]; sent_to?: string[]; keywords: string[]; derived_from: string[]; supersedes?: string[]; superseded?: boolean; tokens?: number; orig_tokens?: number; strength?: number; use_count?: number; n_writers?: number; originals: Original[] }
interface TraceNode { id: string; text: string; agents: string[]; kind: string; turn: number; originals: Original[]; superseded: boolean; sources: TraceNode[] }

function useFetch<T>(url?: string) {
  const [data, setData] = useState<T>();
  const [err, setErr] = useState<string>();
  useEffect(() => {
    setData(undefined); setErr(undefined);
    if (url) getJSON<T>(url).then(setData).catch((e) => setErr(String(e.message ?? e)));
  }, [url]);
  return { data, err };
}

export const AgentTag = ({ name }: { name: string }) => <span className="agent-tag" style={{ color: zoneFor(name).color }}>{zoneFor(name).name}</span>;

export function MemoryDrawer() {
  const { openMemory, runId, set } = useStore();
  const { data, err } = useFetch<MemoryDetail>(openMemory && runId ? `/api/runs/${runId}/memory/${openMemory}` : undefined);
  return (
    <AnimatePresence>
      {openMemory && (
        <>
          <motion.div className="scrim" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={() => set({ openMemory: undefined })} />
          <motion.aside className="drawer" initial={{ x: 460 }} animate={{ x: 0 }} exit={{ x: 460 }} transition={{ type: "spring", stiffness: 320, damping: 34 }}>
            <div className="drawer-head"><div><span className="eyebrow">Memory</span><h2>{openMemory}</h2></div>
              <button className="icon-btn" onClick={() => set({ openMemory: undefined })} aria-label="Close">×</button></div>
            {err && <p className="error">{err}</p>}
            {!data && !err && <p className="muted">Loading…</p>}
            {data && (
              <div className="drawer-body">
                <div className="badges">
                  <span className={`badge ${data.kind}`}>{data.kind}</span>
                  {data.superseded && <span className="badge bad">replaced</span>}
                  {data.strength !== undefined && <span className="badge">strength {data.strength}</span>}
                  {data.n_writers && data.n_writers > 1 && <span className="badge ok">reported {data.n_writers}×</span>}
                </div>
                {(() => {
                  const r = data.kind === "report" ? parseReport(data.text) : undefined;
                  return r
                    ? <><div className="label">Report</div><ReportCard report={r} /></>
                    : <><div className="label">Stored text</div><p className="quote">{data.text}</p></>;
                })()}
                <dl className="dl">
                  <dt>Written by</dt><dd>{data.agents.map((a) => <AgentTag key={a} name={a} />)}</dd>
                  {data.readers && <><dt>Read by</dt><dd>{data.readers.length ? data.readers.map((a) => <AgentTag key={a} name={a} />) : "–"}</dd></>}
                  {data.tokens !== undefined && <><dt>Tokens</dt><dd>{data.tokens} stored · {data.orig_tokens} original</dd></>}
                  {data.derived_from.length > 0 && <><dt>Built on</dt><dd className="chips">{data.derived_from.map((m) => <code key={m} className="chip link" onClick={() => set({ openMemory: m })}>{m}</code>)}</dd></>}
                </dl>
                <div className="label">Keywords</div>
                <div className="chips">{data.keywords.map((k) => <span key={k} className="chip">{k}</span>)}</div>
                <div className="label">Original messages · {data.originals.length}</div>
                {data.originals.map((o, i) => <p key={i} className="original"><AgentTag name={o.agent} /> <span className="muted">t{o.turn}{o.event_id ? ` · ${o.event_id}` : ""}</span><br />{o.text}</p>)}
                <button className="btn primary wide" onClick={() => set({ openTrace: openMemory })}>Show trace</button>
              </div>
            )}
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}

function TraceTree({ node, depth = 0 }: { node: TraceNode; depth?: number }) {
  const set = useStore((s) => s.set);
  return (
    <motion.div className="trace-node" initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: depth * 0.08 }}>
      <div className="trace-head">
        <code className="chip link" onClick={() => set({ openMemory: node.id, openTrace: undefined })}>{node.id}</code>
        <span className={`badge ${node.kind}`}>{node.kind}</span>
        {node.agents.map((a) => <AgentTag key={a} name={a} />)}
        {node.superseded && <span className="badge bad">replaced</span>}
      </div>
      <p className="trace-text">{readable(node.text)}</p>
      {node.kind === "observation" && node.originals.map((o, i) => <p key={i} className="original small">source {o.event_id ?? ""}: {o.text}</p>)}
      {node.sources.length > 0 && <div className="trace-children">{node.sources.map((c) => <TraceTree key={c.id + depth} node={c} depth={depth + 1} />)}</div>}
    </motion.div>
  );
}

export function TraceModal() {
  const { openTrace, runId, set } = useStore();
  const { data, err } = useFetch<TraceNode>(openTrace && runId ? `/api/runs/${runId}/trace/${openTrace}` : undefined);
  return (
    <AnimatePresence>
      {openTrace && (
        <motion.div className="modal-back" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={() => set({ openTrace: undefined })}>
          <motion.div className="modal" initial={{ opacity: 0, y: 24, scale: 0.97 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: 24 }} transition={{ type: "spring", stiffness: 320, damping: 30 }} onClick={(e) => e.stopPropagation()}>
            <div className="drawer-head"><div><span className="eyebrow">Answer trace</span><h2>{openTrace}</h2></div>
              <button className="icon-btn" onClick={() => set({ openTrace: undefined })} aria-label="Close">×</button></div>
            <p className="muted">How this answer was built: each level shows the facts it rests on, who wrote them, and the original messages.</p>
            {err && <p className="error">{err}</p>}
            {!data && !err && <p className="muted">Loading…</p>}
            {data && <TraceTree node={data} />}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
