import { create } from "zustand";

// ---------- server message types ----------
export interface ScenarioInfo { id: string; kind: string; events: number; questions: number; machines: string[]; has_zones: boolean; injected: boolean }
export interface PlantEvent { id: string; turn: number; type: string; source_agent: string; text: string; zone?: string | null; hops?: number }
export interface WriteMsg { turn: number; agent: string; event_id?: string; decision: string; memory_id: string; target?: string | null; superseded?: string | null; notified: string[]; text: string; original: string; kind: string; tokens: number; orig_tokens?: number; keywords: string[]; derived_from: string[]; stats?: Record<string, unknown>; orig_tokens_total: number }
export interface ReadMsg { turn: number; agent: string; query: string; memory_ids: string[]; tokens_used: number; budget?: number; fallback?: string | null; full_history_tokens: number }
export interface LlmMsg { agent: string; turn: number; provider?: string; model?: string; input_tokens: number; output_tokens?: number; latency_s?: number; tool_calls?: number; error?: string | null; fallback_used?: boolean }
export interface Report { machine: string; root_cause: string; action: string; evidence: string[] }
export interface AnswerMsg { event_id: string; question: string; gold: { machine: string; root_cause: string; action: string; supporting_events: string[] }; hops?: number; report: Report; report_memory_id: string; diagnosis_memory_id: string; diagnosis: string; turn: number; correct: boolean }

export type AgentStatus = "idle" | "writing" | "reading" | "thinking";
export interface AgentState { status: AgentStatus; since: number; writes: number; reads: number; calls: number; inputTokens: number; lastText: string; lastDelivered: string[]; lastMemory?: string; provider?: string }

export interface Packet { id: number; from: string; to: string; color: string; born: number; duration: number }
export interface FeedItem { id: number; turn: number; kind: string; agent?: string; text: string; tag?: string; tagColor?: string; memoryId?: string }
export interface TokenPoint { call: number; graph: number; full: number }

type Status = "disconnected" | "connected" | "running" | "done" | "stopped" | "error";

interface State {
  wsStatus: Status;
  error?: string;
  scenarios: ScenarioInfo[];
  presets: string[];
  runId?: string;
  scenarioId?: string;
  condition: string;
  preset: string;
  models: Record<string, { primary: string; skipped: string[] }>;
  zonesInRun: string[];
  totalEvents: number;
  turn: number;
  agents: Record<string, AgentState>;
  packets: Packet[];
  hubPulse: { at: number; color: string };
  feed: FeedItem[];
  tokens: TokenPoint[];
  graphCum: number;
  fullCum: number;
  lastRead: Record<string, ReadMsg>;
  counts: { create: number; merge: number; update: number; append: number; active: number; superseded: number; origTokens: number; storedTokens: number };
  question?: { text: string; eventId: string; zone?: string | null };
  answers: AnswerMsg[];
  selectedAgent?: string;
  openMemory?: string;      // memory id shown in the drawer
  openTrace?: string;       // memory id shown in the trace modal
  set: (p: Partial<State>) => void;
  reset: () => void;
  handle: (m: Record<string, unknown>) => void;
}

let seq = 1;
const now = () => performance.now();
const blankAgent = (): AgentState => ({ status: "idle", since: 0, writes: 0, reads: 0, calls: 0, inputTokens: 0, lastText: "", lastDelivered: [] });
const DECISION_COLOR: Record<string, string> = { create: "#38bdf8", merge: "#4ade80", update: "#f87171", append: "#94a3b8" };
export const decisionColor = (d: string) => DECISION_COLOR[d] ?? "#94a3b8";

const runReset = {
  turn: 0, totalEvents: 0, agents: {} as Record<string, AgentState>, packets: [] as Packet[], feed: [] as FeedItem[],
  tokens: [] as TokenPoint[], graphCum: 0, fullCum: 0, lastRead: {} as Record<string, ReadMsg>,
  counts: { create: 0, merge: 0, update: 0, append: 0, active: 0, superseded: 0, origTokens: 0, storedTokens: 0 },
  question: undefined, answers: [] as AnswerMsg[], openMemory: undefined, openTrace: undefined, error: undefined,
};

export const useStore = create<State>((set, get) => ({
  wsStatus: "disconnected",
  scenarios: [],
  presets: ["mock"],
  condition: "graph",
  preset: "mock",
  models: {},
  zonesInRun: [],
  hubPulse: { at: 0, color: "#22d3ee" },
  ...runReset,
  set: (p) => set(p),
  reset: () => set({ ...runReset }),

  handle: (m) => {
    const s = get();
    const t = now();
    const agentPatch = (name: string, patch: Partial<AgentState>) => ({
      ...s.agents, [name]: { ...(s.agents[name] ?? blankAgent()), ...patch },
    });
    const feed = (item: Omit<FeedItem, "id">) => [{ id: seq++, ...item }, ...get().feed].slice(0, 250);
    const packet = (from: string, to: string, color: string, duration = 900): Packet => ({ id: seq++, from, to, color, born: t, duration });

    switch (m.type) {
      case "run_started": {
        const sc = m.scenario as { events: PlantEvent[]; zones?: Record<string, unknown> };
        set({ ...runReset, runId: m.run_id as string, wsStatus: "running", models: m.models as State["models"],
          zonesInRun: Object.keys(sc.zones ?? {}), totalEvents: sc.events.length, condition: m.condition as string,
          preset: m.preset as string });
        break;
      }
      case "event": {
        const ev = m.event as PlantEvent;
        const patch: Partial<State> = { turn: m.turn as number };
        if (ev.type === "question") {
          patch.question = { text: ev.text, eventId: ev.id, zone: ev.zone };
          patch.feed = feed({ turn: m.turn as number, kind: "question", agent: "diagnosis", text: ev.text, tag: "QUESTION", tagColor: "#f472b6" });
        }
        set(patch);
        break;
      }
      case "write": {
        const w = m as unknown as WriteMsg;
        const st = (w.stats ?? {}) as Record<string, number>;
        const counts = { ...s.counts, [w.decision]: (s.counts[w.decision as keyof typeof s.counts] as number ?? 0) + 1,
          active: st.active_memories ?? s.counts.active, superseded: st.superseded_memories ?? s.counts.superseded,
          origTokens: w.orig_tokens_total ?? s.counts.origTokens, storedTokens: st.stored_tokens_active ?? s.counts.storedTokens };
        const packets = [...s.packets, packet(w.agent, "hub", decisionColor(w.decision))];
        for (const n of w.notified.slice(0, 3)) packets.push({ ...packet("hub", n, "#22d3ee", 1100), born: t + 700 });
        set({
          counts, packets, hubPulse: { at: t + 800, color: decisionColor(w.decision) },
          agents: agentPatch(w.agent, { status: "writing", since: t, writes: (s.agents[w.agent]?.writes ?? 0) + 1, lastText: w.text, lastMemory: w.memory_id }),
          feed: feed({ turn: w.turn, kind: "write", agent: w.agent, text: w.text, tag: w.decision.toUpperCase(), tagColor: decisionColor(w.decision), memoryId: w.memory_id }),
        });
        break;
      }
      case "read": {
        const r = m as unknown as ReadMsg;
        const packets = [...s.packets];
        r.memory_ids.slice(0, 5).forEach((_, i) => packets.push({ ...packet("hub", r.agent, "#fbbf24", 1000), born: t + i * 120 }));
        set({
          packets, lastRead: { ...s.lastRead, [r.agent]: r },
          agents: agentPatch(r.agent, { status: "reading", since: t, reads: (s.agents[r.agent]?.reads ?? 0) + 1, lastDelivered: r.memory_ids }),
          feed: feed({ turn: r.turn, kind: "read", agent: r.agent, text: `${r.memory_ids.length} facts delivered, ${r.tokens_used} tokens${r.fallback ? ` (${r.fallback} fallback)` : ""}`, tag: "READ", tagColor: "#fbbf24" }),
        });
        break;
      }
      case "llm_start": {
        const a = m.agent as string;
        set({ agents: agentPatch(a, { status: "thinking", since: t, provider: m.provider as string }) });
        break;
      }
      case "llm": {
        const l = m as unknown as LlmMsg;
        const rd = s.lastRead[l.agent];
        const fullCall = rd ? l.input_tokens - rd.tokens_used + rd.full_history_tokens : l.input_tokens;
        const graphCum = s.graphCum + l.input_tokens;
        const fullCum = s.fullCum + Math.max(fullCall, l.input_tokens);
        const a = s.agents[l.agent] ?? blankAgent();
        set({
          graphCum, fullCum, tokens: [...s.tokens, { call: s.tokens.length + 1, graph: graphCum, full: fullCum }],
          agents: agentPatch(l.agent, { status: "idle", since: t, calls: a.calls + 1, inputTokens: a.inputTokens + l.input_tokens, provider: l.provider }),
          feed: feed({ turn: l.turn, kind: "llm", agent: l.agent, text: l.error ? `error: ${l.error}` :
              `${l.provider ?? "?"} · ${l.input_tokens} in / ${l.output_tokens ?? 0} out · ${(l.latency_s ?? 0).toFixed(2)}s${l.tool_calls ? " · tool call" : ""}`,
            tag: l.error ? "ERROR" : "LLM", tagColor: l.error ? "#f87171" : "#a78bfa" }),
        });
        break;
      }
      case "maintain":
        set({ feed: feed({ turn: s.turn, kind: "maintain", text: `maintenance: ${m.archived} archived, ${m.keywords_merged} keywords merged`, tag: "MAINT", tagColor: "#64748b" }) });
        break;
      case "answer": {
        const a = m as unknown as AnswerMsg;
        set({ answers: [...s.answers, a], question: undefined,
          feed: feed({ turn: a.turn, kind: "answer", agent: "report", text: `${a.report.machine}: ${a.report.root_cause} (expected ${a.gold.root_cause})`,
            tag: a.correct ? "CORRECT" : "WRONG", tagColor: a.correct ? "#4ade80" : "#f87171", memoryId: a.report_memory_id }) });
        break;
      }
      case "done":
        set({ wsStatus: (m.status as Status) ?? "done" });
        break;
      case "error":
        set({ wsStatus: "error", error: m.message as string });
        break;
    }
  },
}));

// Agents fall back to idle a moment after their last action, so the floor calms down between events.
export function settleAgents() {
  const s = useStore.getState();
  const t = now();
  let changed = false;
  const agents = { ...s.agents };
  for (const [k, a] of Object.entries(agents)) {
    if (a.status !== "idle" && a.status !== "thinking" && t - a.since > 1600) { agents[k] = { ...a, status: "idle" }; changed = true; }
  }
  const packets = s.packets.filter((p) => t - p.born < p.duration + 100);
  if (changed || packets.length !== s.packets.length) s.set({ agents, packets });
}
