import { AnimatePresence, motion } from "framer-motion";
import { ZONE_BY_ID, zoneFor } from "../layout";
import { STATUS_COLOR, STATUS_LABEL } from "../status";
import { decisionColor, useStore } from "../store";
import { readable } from "../report";
import { ReportCard } from "./ReportCard";

const ROLE_OF = (id: string) => (["history", "diagnosis", "report"].includes(id) ? id : "sensor");

export function AgentSheet({ id, onClose }: { id: string; onClose?: () => void }) {
  const { agents, models, commits, memories, answers, set } = useStore();
  const mine = commits[id] ?? [];
  const z = ZONE_BY_ID[id] ?? zoneFor(id);
  const a = agents[id];
  const model = models[ROLE_OF(id)];
  const status = a?.status ?? "idle";
  return (
    <motion.div className="sheet" style={{ ["--accent" as string]: z.color }} initial={{ opacity: 0, x: 24 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 24 }} transition={{ type: "spring", stiffness: 300, damping: 30 }}>
      <div className="sheet-head">
        <div>
          <span className="eyebrow">{z.kind === "support" ? "Support agent" : "Production zone"}</span>
          <h3>{z.name}</h3>
          <p className="muted">{z.subtitle}</p>
        </div>
        {onClose && <button className="icon-btn" onClick={onClose} aria-label="Close">×</button>}
      </div>
      <div className="pill" style={{ ["--c" as string]: STATUS_COLOR[status] }}><i />{STATUS_LABEL[status]}</div>
      <div className="mini-stats">
        <div><b>{a?.writes ?? 0}</b><span>writes</span></div>
        <div><b>{a?.reads ?? 0}</b><span>reads</span></div>
        <div><b>{a?.calls ?? 0}</b><span>LLM calls</span></div>
        <div><b>{(a?.inputTokens ?? 0).toLocaleString()}</b><span>input tokens</span></div>
      </div>
      <dl className="dl"><dt>Model</dt><dd>{model?.primary ?? "Not used in this run"}</dd></dl>
      {id === "report" && (
        <div className="block">
          <div className="label">Reports written � {answers.length}</div>
          {answers.length === 0 ? <p className="muted small">No report yet. One is written after each diagnosis question.</p> : (
            <div className="rlist">
              {[...answers].reverse().map((a) => (
                <ReportCard key={a.event_id} report={{ ...a.report }} question={a.question} verdict={a.correct}
                  expected={a.gold} traceId={a.report_memory_id} />
              ))}
            </div>
          )}
        </div>
      )}
      <div className="block">
        <div className="label">Commits · {mine.length}</div>
        {mine.length === 0 ? <p className="muted small">Nothing committed yet.</p> : (
          <ul className="commits">
            <AnimatePresence initial={false}>
              {[...mine].reverse().slice(0, 8).map((c) => (
                <motion.li key={c.seq} initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} onClick={() => set({ openMemory: c.memoryId })}>
                  <div className="c-head">
                    <span className="ftag" style={{ color: decisionColor(c.decision) }}>{c.decision.toUpperCase()}</span>
                    <code>{c.memoryId}</code>
                    {c.superseded && <span className="c-note">replaces {c.superseded}</span>}
                    {memories[c.memoryId]?.supersededBy && <span className="c-note bad">replaced</span>}
                    <span className="c-turn">t{c.turn}</span>
                  </div>
                  <p>{readable(c.text)}</p>
                  {c.notified.length > 0 && <div className="c-to">→ {c.notified.map((n) => zoneFor(n).name).join(", ")}</div>}
                </motion.li>
              ))}
            </AnimatePresence>
          </ul>
        )}
      </div>
      {a?.lastDelivered?.length ? (
        <div className="block">
          <div className="label">Last facts received</div>
          <div className="chips">{a.lastDelivered.map((m) => <code key={m} className="chip link" onClick={() => set({ openMemory: m })}>{m}</code>)}</div>
        </div>
      ) : null}
    </motion.div>
  );
}
