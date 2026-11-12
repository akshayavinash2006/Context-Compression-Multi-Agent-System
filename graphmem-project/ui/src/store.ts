import { create } from "zustand";
import { readable } from "./report";

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
// Agent -> agent context transfer: a memory written by `from` reached `to` (pushed, read, or handed off).
export type FlowVia = "push" | "read" | "handoff";
export interface ContextFlow { from: string; to: string; memories: string[]; count: number; tokens: number; lastTurn: number; via: FlowVia[] }

// Everything the UI has seen about one memory, built from the live stream.
export interface MemInfo {
  id: string; text: string; kind: string; turn: number; tokens: number; origTokens?: number;
  authors: string[]; readers: string[]; pushedTo: string[]; derivedFrom: string[];
  supersedes?: string; supersededBy?: string; keywords: string[];
}
// One write by one agent. Merges commit into an existing memory, so one memory can have several commits.
export interface Commit { seq: number; memoryId: string; decision: string; turn: number; text: string; original: string; notified: string[]; superseded?: string | null }

const MAX_COMMITS = 60;
const addUnique = (xs: string[], x: string) => (xs.includes(x) ? xs : [...xs, x]);

function addFlow(flows: ContextFlow[], from: string, to: string, memoryId: string, via: FlowVia, turn: number, tokens = 0): ContextFlow[] {
  if (!from || !to || from === to) return flows;
  const i = flows.findIndex((f) => f.from === from && f.to === to);
  if (i < 0) return [...flows, { from, to, memories: [memoryId], count: 1, tokens, lastTurn: turn, via: [via] }];
  const f = flows[i];
  const next = { ...f, memories: addUnique(f.memories, memoryId).slice(-24), count: f.count + 1, lastTurn: turn,
    tokens: tokens || f.tokens, via: addUnique(f.via, via) as FlowVia[] };
  return flows.map((g, j) => (j === i ? next : g));
}

export interface RunConfig { scenario: string; preset: string; condition: string; mode: string; delay: number }

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
  contextFlows: ContextFlow[];
  memories: Record<string, MemInfo>;
  commits: Record<string, Commit[]>;   // agent -> its writes, oldest first
  graphCum: number;
  fullCum: number;
  lastRead: Record<string, ReadMsg>;
  counts: { create: number; merge: number; update: number; append: number; active: number; superseded: number; origTokens: number; storedTokens: number };
  question?: { text: string; eventId: string; zone?: string | null };
  answers: AnswerMsg[];
  starting: boolean;        // Start pressed, waiting for the server to prepare the run
  startedAt?: number;       // performance.now() when the run began
  finishedAt?: number;
  cfg: RunConfig;
  setCfg: (p: Partial<RunConfig>) => void;
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
  tokens: [] as TokenPoint[], contextFlows: [] as ContextFlow[], memories: {} as Record<string, MemInfo>, commits: {} as Record<string, Commit[]>, graphCum: 0, fullCum: 0, lastRead: {} as Record<string, ReadMsg>,
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
  starting: false,
  cfg: { scenario: "", preset: "mock", condition: "graph", mode: "replay", delay: 0.5 },
  setCfg: (p) => set({ cfg: { ...get().cfg, ...p } }),
  set: (p) => set(p),
  reset: () => set({ ...runReset }),

  handle: (m) => {
    const s = get();
    // A run that was replaced or stopped can still deliver late messages; ignore anything from another run.
    if (m.type !== "run_started" && m.run_id && s.runId && m.run_id !== s.runId) return;
    const t = now();
    const agentPatch = (name: string, patch: Partial<AgentState>) => ({
      ...s.agents, [name]: { ...(s.agents[name] ?? blankAgent()), ...patch },
    });
    const feed = (item: Omit<FeedItem, "id">) => [{ id: seq++, ...item }, ...get().feed].slice(0, 250);
    const packet = (from: string, to: string, color: string, duration = 900): Packet => ({ id: seq++, from, to, color, born: t, duration });

    switch (m.type) {
      case "run_started": {
        const sc = m.scenario as { events: PlantEvent[]; zones?: Record<string, unknown> };
        set({ ...runReset, starting: false, startedAt: t, finishedAt: undefined, runId: m.run_id as string, wsStatus: "running", models: m.models as State["models"],
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
        const w0 = m as unknown as WriteMsg;
        const w: WriteMsg = { ...w0, text: readable(w0.text) };   // reports are stored as JSON; show them as a sentence
        const st = (w.stats ?? {}) as Record<string, number>;
        const counts = { ...s.counts, [w.decision]: (s.counts[w.decision as keyof typeof s.counts] as number ?? 0) + 1,
          active: st.active_memories ?? s.counts.active, superseded: st.superseded_memories ?? s.counts.superseded,
          origTokens: w.orig_tokens_total ?? s.counts.origTokens, storedTokens: st.stored_tokens_active ?? s.counts.storedTokens };
        const packets = [...s.packets, packet(w.agent, "hub", decisionColor(w.decision))];
        for (const n of w.notified.slice(0, 3)) packets.push({ ...packet("hub", n, "#22d3ee", 1100), born: t + 700 });

        const prev = s.memories[w.memory_id];
        const memories = { ...s.memories, [w.memory_id]: {
          id: w.memory_id, text: w.text, kind: w.kind, turn: prev?.turn ?? w.turn, tokens: w.tokens, origTokens: w.orig_tokens,
          authors: addUnique(prev?.authors ?? [], w.agent), readers: prev?.readers ?? [],
          pushedTo: [...new Set([...(prev?.pushedTo ?? []), ...w.notified])], derivedFrom: w.derived_from ?? [],
          keywords: w.keywords ?? [], supersedes: w.superseded ?? prev?.supersedes, supersededBy: prev?.supersededBy,
        } };
        if (w.superseded && memories[w.superseded]) memories[w.superseded] = { ...memories[w.superseded], supersededBy: w.memory_id };
        const commit: Commit = { seq: seq++, memoryId: w.memory_id, decision: w.decision, turn: w.turn, text: w.text,
          original: w.original, notified: w.notified, superseded: w.superseded };
        const commits = { ...s.commits, [w.agent]: [...(s.commits[w.agent] ?? []), commit].slice(-MAX_COMMITS) };
        let flows = s.contextFlows;
        for (const n of w.notified) flows = addFlow(flows, w.agent, n, w.memory_id, "push", w.turn);

        set({
          counts, packets, memories, commits, contextFlows: flows, hubPulse: { at: t + 800, color: decisionColor(w.decision) },
          agents: agentPatch(w.agent, { status: "writing", since: t, writes: (s.agents[w.agent]?.writes ?? 0) + 1, lastText: w.text, lastMemory: w.memory_id }),
          feed: feed({ turn: w.turn, kind: "write", agent: w.agent, text: w.text, tag: w.decision.toUpperCase(), tagColor: decisionColor(w.decision), memoryId: w.memory_id }),
        });
        break;
      }
      case "read": {
        const r = m as unknown as ReadMsg;
        const packets = [...s.packets];
        r.memory_ids.slice(0, 5).forEach((_, i) => packets.push({ ...packet("hub", r.agent, "#fbbf24", 1000), born: t + i * 120 }));
        const memories = { ...s.memories };
        let flows = s.contextFlows;
        for (const id of r.memory_ids) {
          const mi = memories[id];
          if (!mi) continue;
          memories[id] = { ...mi, readers: addUnique(mi.readers, r.agent) };
          for (const author of mi.authors) flows = addFlow(flows, author, r.agent, id, "read", r.turn, r.tokens_used);
        }
        set({
          packets, memories, contextFlows: flows, lastRead: { ...s.lastRead, [r.agent]: r },
          agents: agentPatch(r.agent, { status: "reading", since: t, reads: (s.agents[r.agent]?.reads ?? 0) + 1, lastDelivered: r.memory_ids }),
          feed: feed({ turn: r.turn, kind: "read", agent: r.agent, text: `${r.memory_ids.length} facts delivered, ${r.tokens_used} tokens${r.fallback ? ` (${r.fallback} fallback)` : ""}`, tag: "READ", tagColor: "#fbbf24" }),
        });
        break;
      }
      case "context_flow": {
        // Explicit hand-off (Diagnosis -> Report): the memory goes straight into the receiver's prompt.
        const from = m.from as string, to = m.to as string;
        let flows = s.contextFlows;
        const memories = { ...s.memories };
        for (const id of (m.memory_ids as string[] | undefined) ?? []) {
          flows = addFlow(flows, from, to, id, "handoff", m.turn as number, m.tokens as number);
          if (memories[id]) memories[id] = { ...memories[id], readers: addUnique(memories[id].readers, to) };
        }
        set({ contextFlows: flows, memories });
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
        set({ wsStatus: (m.status as Status) ?? "done", finishedAt: t, starting: false });
        break;
      case "error":
        set({ wsStatus: "error", error: m.message as string, starting: false });
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
