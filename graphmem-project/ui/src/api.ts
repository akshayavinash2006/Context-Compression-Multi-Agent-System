import { useStore } from "./store";

let ws: WebSocket | null = null;
let retry: number | undefined;

export function connect() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/run`);
  ws.onopen = () => {
    useStore.getState().set({ wsStatus: "connected", error: undefined });
    // The server may have started after the page loaded, so fetch scenarios and presets again.
    if (useStore.getState().scenarios.length === 0) void loadInitial();
  };
  ws.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (m.type !== "hello") useStore.getState().handle(m);
  };
  ws.onclose = () => {
    useStore.getState().set({ wsStatus: "disconnected" });
    window.clearTimeout(retry);
    retry = window.setTimeout(connect, 2000);
  };
}

export interface StartOptions { scenario: string; preset: string; condition: string; mode: string; delay: number }

export function startRun(o: StartOptions) {
  const st = useStore.getState();
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    st.set({ error: "Cannot start: the backend is not connected. Check that the server is running on port 8000." });
    return;
  }
  st.reset();
  st.set({ starting: true, startedAt: performance.now(), finishedAt: undefined, scenarioId: o.scenario, wsStatus: "connected" });
  ws.send(JSON.stringify({ type: "start", ...o }));
}

export function stopRun() {
  ws?.send(JSON.stringify({ type: "stop" }));
}

export async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}

export async function loadInitial() {
  try {
    const [meta, scenarios] = await Promise.all([
      getJSON<{ presets: string[]; default_preset: string }>("/api/meta"),
      getJSON<import("./store").ScenarioInfo[]>("/api/scenarios"),
    ]);
    useStore.getState().set({ presets: meta.presets, scenarios });
  } catch (e) {
    useStore.getState().set({ error: `Server not reachable: ${(e as Error).message}` });
  }
}

export async function generateScenarios(n = 10) {
  await fetch(`/api/scenarios/generate?n=${n}`, { method: "POST" });
  await loadInitial();
}
