import { useEffect, useState } from "react";
import { getJSON } from "../api";
import { zoneFor } from "../layout";
import { useStore } from "../store";

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

const Agent = ({ name }: { name: string }) => <span className="agent-tag" style={{ color: zoneFor(name).color }}>{zoneFor(name).name}</span>;

export function MemoryDrawer() {
  const { openMemory, runId, set } = useStore();
  const { data, err } = useFetch<MemoryDetail>(openMemory && runId ? `/api/runs/${runId}/memory/${openMemory}` : undefined);
  if (!openMemory) return null;
  return (
    <aside className="drawer">
      <div className="card-head"><h2>Memory {openMemory}</h2><button className="ghost small" onClick={() => set({ openMemory: undefined })}>Close</button></div>
      {err && <p className="error">{err}</p>}
      {!data && !err && <p className="empty">Loading…</p>}
      {data && (
        <>
          <div className="mem-badges">
            <span className={`badge ${data.kind}`}>{data.kind}</span>
            {data.superseded && <span className="badge bad">replaced</span>}
            {data.strength !== undefined && <span className="badge">strength {data.strength}</span>}
            {data.n_writers && data.n_writers > 1 && <span className="badge ok">reported {data.n_writers}×</span>}
          </div>
          <div className="block-label">Stored (compressed) text</div>
          <p className="mem-text">{data.text}</p>
          <dl className="agent-dl">
            <dt>Written by</dt><dd>{data.agents.map((a) => <Agent key={a} name={a} />)}</dd>
            {data.readers && <><dt>Read by</dt><dd>{data.readers.length ? data.readers.map((a) => <Agent key={a} name={a} />) : "–"}</dd></>}
            {data.tokens !== undefined && <><dt>Tokens</dt><dd>{data.tokens} stored / {data.orig_tokens} original</dd></>}
            {data.derived_from.length > 0 && <><dt>Built on</dt><dd className="chips">{data.derived_from.map((m) => <code key={m} className="clickable" onClick={() => set({ openMemory: m })}>{m}</code>)}</dd></>}
          </dl>
          <div className="block-label">Keywords</div>
          <div className="chips">{data.keywords.map((k) => <span key={k} className="kw">{k}</span>)}</div>
          <div className="block-label">Original messages ({data.originals.length})</div>
          {data.originals.map((o, i) => <p key={i} className="original"><Agent name={o.agent} /> t{o.turn}{o.event_id ? ` · ${o.event_id}` : ""}<br />{o.text}</p>)}
          <button className="primary" onClick={() => set({ openTrace: openMemory })}>Show trace</button>
        </>
      )}
    </aside>
  );
}

function TraceTree({ node, depth = 0 }: { node: TraceNode; depth?: number }) {
  const set = useStore((s) => s.set);
  return (
    <div className={`trace-node depth-${Math.min(depth, 3)}`}>
      <div className="trace-head">
        <code className="clickable" onClick={() => set({ openMemory: node.id, openTrace: undefined })}>{node.id}</code>
        <span className={`badge ${node.kind}`}>{node.kind}</span>
        {node.agents.map((a) => <Agent key={a} name={a} />)}
        {node.superseded && <span className="badge bad">replaced</span>}
      </div>
      <p className="trace-text">{node.text}</p>
      {node.kind === "observation" && node.originals.map((o, i) => <p key={i} className="original small">source {o.event_id ?? ""}: {o.text}</p>)}
      {node.sources.length > 0 && <div className="trace-children">{node.sources.map((c) => <TraceTree key={c.id + depth} node={c} depth={depth + 1} />)}</div>}
    </div>
  );
}

export function TraceModal() {
  const { openTrace, runId, set } = useStore();
  const { data, err } = useFetch<TraceNode>(openTrace && runId ? `/api/runs/${runId}/trace/${openTrace}` : undefined);
  if (!openTrace) return null;
  return (
    <div className="modal-backdrop" onClick={() => set({ openTrace: undefined })}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="card-head"><h2>Trace of {openTrace}</h2><button className="ghost small" onClick={() => set({ openTrace: undefined })}>Close</button></div>
        <p className="muted-text">How this answer was built: each level shows the facts it rests on, who wrote them, and the original messages.</p>
        {err && <p className="error">{err}</p>}
        {!data && !err && <p className="empty">Loading…</p>}
        {data && <TraceTree node={data} />}
      </div>
    </div>
  );
}
