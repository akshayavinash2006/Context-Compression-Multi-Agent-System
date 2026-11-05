import { motion } from "framer-motion";
import { ZONE_BY_ID, zoneFor } from "../layout";
import { STATUS_COLOR, STATUS_LABEL } from "../status";
import { useStore } from "../store";

const ROLE_OF = (id: string) => (["history", "diagnosis", "report"].includes(id) ? id : "sensor");

export function AgentSheet({ id, onClose }: { id: string; onClose?: () => void }) {
  const { agents, models, set } = useStore();
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
      {a?.lastText && (
        <div className="block">
          <div className="label">Last fact written {a.lastMemory && <code className="chip link" onClick={() => set({ openMemory: a.lastMemory })}>{a.lastMemory}</code>}</div>
          <p className="quote">{a.lastText}</p>
        </div>
      )}
      {a?.lastDelivered?.length ? (
        <div className="block">
          <div className="label">Last facts received</div>
          <div className="chips">{a.lastDelivered.map((m) => <code key={m} className="chip link" onClick={() => set({ openMemory: m })}>{m}</code>)}</div>
        </div>
      ) : null}
    </motion.div>
  );
}
