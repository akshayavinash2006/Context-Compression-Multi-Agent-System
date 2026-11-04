import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { generateScenarios, startRun, stopRun } from "../api";
import { ROUTES, type Route } from "../router";
import { useStore } from "../store";

const STATUS_TEXT: Record<string, string> = {
  disconnected: "Server offline", connected: "Ready", running: "Live", done: "Finished", stopped: "Stopped", error: "Error",
};

function Select({ label, value, onChange, disabled, children }: { label: string; value: string; onChange: (v: string) => void; disabled?: boolean; children: React.ReactNode }) {
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value} disabled={disabled} onChange={(e) => onChange(e.target.value)}>{children}</select>
    </label>
  );
}

function Settings({ onClose }: { onClose: () => void }) {
  const { cfg, setCfg, scenarios, presets, wsStatus } = useStore();
  const plant = scenarios.filter((s) => s.has_zones);
  const running = wsStatus === "running";
  return (
    <motion.div className="popover" initial={{ opacity: 0, y: -8, scale: 0.97 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: -8, scale: 0.97 }} transition={{ duration: 0.16 }}>
      <div className="popover-head"><b>Run settings</b><button className="icon-btn" onClick={onClose} aria-label="Close">×</button></div>
      <Select label="Scenario" value={cfg.scenario} disabled={running} onChange={(v) => setCfg({ scenario: v })}>
        {plant.length === 0 && <option value="">No scenarios yet</option>}
        {plant.map((s) => <option key={s.id} value={s.id}>{s.id} · {s.events} events · {s.questions} questions</option>)}
      </Select>
      {plant.length === 0 && <button className="btn ghost" onClick={() => generateScenarios(10)}>Generate sample scenarios</button>}
      <div className="field-row">
        <Select label="Models" value={cfg.preset} disabled={running} onChange={(v) => setCfg({ preset: v })}>
          {presets.map((p) => <option key={p} value={p}>{p}</option>)}
        </Select>
        <Select label="Memory" value={cfg.condition} disabled={running} onChange={(v) => setCfg({ condition: v })}>
          <option value="graph">Graph memory</option>
          <option value="full">Full history</option>
          <option value="vector">Vector RAG</option>
          <option value="window">Last 10 messages</option>
          <option value="summary">Rolling summary</option>
        </Select>
      </div>
      <Select label="Mode" value={cfg.mode} disabled={running} onChange={(v) => setCfg({ mode: v })}>
        <option value="replay">Replay · only Diagnosis and Report call LLMs</option>
        <option value="live">Live · every agent calls an LLM</option>
      </Select>
      <label className="field">
        <span>Pace <em>{cfg.delay.toFixed(1)}s per event</em></span>
        <input type="range" min={0} max={2} step={0.1} value={cfg.delay} disabled={running} onChange={(e) => setCfg({ delay: +e.target.value })} />
      </label>
    </motion.div>
  );
}

export function TopBar({ route }: { route: Route }) {
  const { wsStatus, turn, totalEvents, cfg, setCfg, scenarios } = useStore();
  const [open, setOpen] = useState(false);
  const plant = scenarios.filter((s) => s.has_zones);
  useEffect(() => { if (!cfg.scenario && plant.length) setCfg({ scenario: plant[0].id }); }, [plant, cfg.scenario, setCfg]);

  const running = wsStatus === "running";
  const canStart = wsStatus !== "disconnected" && !!cfg.scenario && !running;
  const pct = totalEvents ? Math.round((turn / totalEvents) * 100) : 0;
  const meta = ROUTES.find((r) => r.id === route)!;

  return (
    <header className="topbar">
      <div className="crumb">
        <span className="crumb-title">{meta.label}</span>
        <span className="crumb-hint">{meta.hint}</span>
      </div>
      <div className="topbar-right">
        <div className={`status ${wsStatus}`}>
          <i />{STATUS_TEXT[wsStatus]}
          {totalEvents > 0 && <em>{turn}/{totalEvents}</em>}
        </div>
        {totalEvents > 0 && (
          <div className="progress" aria-label={`${pct}% of events processed`}><motion.span animate={{ width: `${pct}%` }} transition={{ ease: "easeOut", duration: 0.4 }} /></div>
        )}
        <div className="settings-anchor">
          <button className="btn ghost icon" onClick={() => setOpen((o) => !o)} aria-label="Run settings" aria-expanded={open}>
            <svg viewBox="0 0 24 24"><path d="M4 6h10M18 6h2M4 12h2M10 12h10M4 18h12M20 18h0" /><circle cx="16" cy="6" r="2" /><circle cx="8" cy="12" r="2" /><circle cx="18" cy="18" r="2" /></svg>
          </button>
          <AnimatePresence>{open && <Settings onClose={() => setOpen(false)} />}</AnimatePresence>
        </div>
        {running
          ? <button className="btn danger" onClick={stopRun}>Stop</button>
          : <motion.button className="btn primary" disabled={!canStart} whileTap={{ scale: 0.97 }} onClick={() => startRun(cfg)}>
              <svg viewBox="0 0 24 24"><path d="M7 5v14l12-7z" /></svg>Start run
            </motion.button>}
      </div>
    </header>
  );
}
