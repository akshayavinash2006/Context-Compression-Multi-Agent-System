import { ZONE_BY_ID, zoneFor } from "../layout";
import { STATUS_COLOR } from "../scene/Zone";
import { useStore } from "../store";

const ROLE_OF = (id: string) => (["history", "diagnosis", "report"].includes(id) ? id : "sensor");

export function AgentPanel() {
  const { selectedAgent, agents, models, set } = useStore();
  if (!selectedAgent) {
    return (
      <section className="card agent">
        <div className="card-head"><h2>Agent</h2></div>
        <p className="empty">Click a zone on the plant to see its agent.</p>
      </section>
    );
  }
  const z = ZONE_BY_ID[selectedAgent] ?? zoneFor(selectedAgent);
  const a = agents[selectedAgent];
  const model = models[ROLE_OF(selectedAgent)];
  return (
    <section className="card agent" style={{ ["--accent" as string]: z.color }}>
      <div className="card-head">
        <h2><span className="dot" style={{ background: STATUS_COLOR[a?.status ?? "idle"] }} />{z.name} agent</h2>
        <button className="ghost small" onClick={() => set({ selectedAgent: undefined })}>Close</button>
      </div>
      <div className="agent-sub">{z.subtitle}</div>
      <dl className="agent-dl">
        <dt>Model</dt><dd>{model?.primary ?? "not used in this run"}</dd>
        <dt>Writes / reads</dt><dd>{a?.writes ?? 0} / {a?.reads ?? 0}</dd>
        <dt>LLM calls</dt><dd>{a?.calls ?? 0} · {(a?.inputTokens ?? 0).toLocaleString()} input tokens</dd>
      </dl>
      {a?.lastText && (
        <div className="agent-block">
          <div className="block-label">Last fact written {a.lastMemory && <code className="clickable" onClick={() => set({ openMemory: a.lastMemory })}>{a.lastMemory}</code>}</div>
          <p>{a.lastText}</p>
        </div>
      )}
      {a?.lastDelivered?.length ? (
        <div className="agent-block">
          <div className="block-label">Last facts received from memory</div>
          <div className="chips">{a.lastDelivered.map((m) => <code key={m} className="clickable" onClick={() => set({ openMemory: m })}>{m}</code>)}</div>
        </div>
      ) : null}
    </section>
  );
}
