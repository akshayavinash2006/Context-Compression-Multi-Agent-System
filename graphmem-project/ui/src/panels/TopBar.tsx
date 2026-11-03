import { useEffect, useState } from "react";
import { generateScenarios, startRun, stopRun } from "../api";
import { useStore } from "../store";

const STATUS_TEXT: Record<string, string> = {
  disconnected: "Server offline", connected: "Ready", running: "Running", done: "Finished", stopped: "Stopped", error: "Error",
};

export function TopBar() {
  const { scenarios, presets, wsStatus, turn, totalEvents } = useStore();
  const plant = scenarios.filter((s) => s.has_zones);
  const [scenario, setScenario] = useState("");
  const [preset, setPreset] = useState("mock");
  const [condition, setCondition] = useState("graph");
  const [mode, setMode] = useState("replay");
  const [delay, setDelay] = useState(0.5);

  useEffect(() => { if (!scenario && plant.length) setScenario(plant[0].id); }, [plant, scenario]);

  const running = wsStatus === "running";
  const canStart = wsStatus !== "disconnected" && !!scenario && !running;

  return (
    <header className="topbar">
      <div className="brand">AUTO<span>PLANT</span><em>GraphMem</em></div>
      <div className="controls">
        <label>Scenario
          <select value={scenario} onChange={(e) => setScenario(e.target.value)} disabled={running}>
            {plant.length === 0 && <option value="">none</option>}
            {plant.map((s) => <option key={s.id} value={s.id}>{s.id} · {s.events} events · {s.questions} questions</option>)}
          </select>
        </label>
        {plant.length === 0 && <button className="ghost" onClick={() => generateScenarios(10)}>Generate scenarios</button>}
        <label>Models
          <select value={preset} onChange={(e) => setPreset(e.target.value)} disabled={running}>
            {presets.map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
        </label>
        <label>Memory
          <select value={condition} onChange={(e) => setCondition(e.target.value)} disabled={running}>
            <option value="graph">Graph memory</option>
            <option value="full">Full history</option>
            <option value="vector">Vector RAG</option>
            <option value="window">Last 10 messages</option>
            <option value="summary">Rolling summary</option>
          </select>
        </label>
        <label>Mode
          <select value={mode} onChange={(e) => setMode(e.target.value)} disabled={running}>
            <option value="replay">Replay (only Diagnosis &amp; Report call LLMs)</option>
            <option value="live">Live (every agent calls an LLM)</option>
          </select>
        </label>
        <label className="speed">Pace {delay.toFixed(1)}s
          <input type="range" min={0} max={2} step={0.1} value={delay} onChange={(e) => setDelay(+e.target.value)} disabled={running} />
        </label>
        {running
          ? <button className="danger" onClick={stopRun}>Stop</button>
          : <button className="primary" disabled={!canStart} onClick={() => startRun({ scenario, preset, condition, mode, delay })}>Start run</button>}
      </div>
      <div className="status">
        <span className={`pill ${wsStatus}`}>{STATUS_TEXT[wsStatus]}</span>
        {totalEvents > 0 && <span className="turn">Turn {turn} / {totalEvents}</span>}
      </div>
    </header>
  );
}
