import { useEffect } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { connect, loadInitial, startRun } from "./api";
import { MemoryDrawer, TraceModal } from "./components/Overlays";
import { TopBar } from "./components/TopBar";
import { Activity } from "./pages/Activity";
import { Agents } from "./pages/Agents";
import { Analytics } from "./pages/Analytics";
import { Diagnosis } from "./pages/Diagnosis";
import { MemoryGraph } from "./pages/MemoryGraph";
import { Overview } from "./pages/Overview";
import { Run } from "./pages/Run";
import { ROUTES, go, useRoute, type Route } from "./router";
import { settleAgents, useStore } from "./store";

let autostarted = false;

const ICONS: Record<Route, string> = {
  run: "M7 4.5v15l12-7.5z",
  overview: "M3 21V9l6 3V9l6 3V5l6 4v12z",
  analytics: "M4 20V10m6 10V4m6 16v-7m5 7H3",
  activity: "M3 12h4l3-8 4 16 3-8h4",
  diagnosis: "M9 3h6v3l4 4v11H5V10l4-4zM9 14h6m-3-3v6",
  graph: "M6 6a2 2 0 100 .01M18 8a2 2 0 100 .01M12 18a2 2 0 100 .01M7.7 7.2l3 9m6-7.8l-4 8m-4.7-10l9 1.2",
  agents: "M12 3l8 4.5v9L12 21l-8-4.5v-9zM12 12l8-4.5M12 12v9M12 12L4 7.5",
};

const PAGES: Record<Route, () => React.JSX.Element> = {
  overview: Overview, run: Run, analytics: Analytics, activity: Activity, diagnosis: Diagnosis, graph: MemoryGraph, agents: Agents,
};

export default function App() {
  const route = useRoute();
  const error = useStore((s) => s.error);
  const wsStatus = useStore((s) => s.wsStatus);
  const running = wsStatus === "running";

  useEffect(() => {
    connect();
    loadInitial();
    const id = window.setInterval(settleAgents, 300);
    return () => window.clearInterval(id);
  }, []);

  // Demo link: /?autostart=P001&preset=mock&delay=0.5 starts a run as soon as the server is connected.
  useEffect(() => {
    const q = new URLSearchParams(location.search);
    const sc = q.get("autostart");
    if (sc && wsStatus === "connected" && !autostarted) {
      autostarted = true;
      startRun({ scenario: sc, preset: q.get("preset") ?? "mock", condition: q.get("condition") ?? "graph",
        mode: q.get("mode") ?? "replay", delay: Number(q.get("delay") ?? 0.5) });
    }
  }, [wsStatus]);

  const Page = PAGES[route];
  return (
    <div className="app">
      <div className="bg-glow" aria-hidden />
      <nav className="sidebar" aria-label="Main">
        <div className="logo" aria-label="AutoPlant"><span /></div>
        <div className="nav">
          {ROUTES.map((r) => (
            <button key={r.id} className={`nav-item ${route === r.id ? "active" : ""}`} onClick={() => go(r.id)} aria-label={r.label} aria-current={route === r.id ? "page" : undefined}>
              {route === r.id && <motion.i layoutId="nav-pill" className="nav-pill" transition={{ type: "spring", stiffness: 420, damping: 34 }} />}
              <svg viewBox="0 0 24 24"><path d={ICONS[r.id]} /></svg>
              <span className="nav-tip">{r.label}</span>
            </button>
          ))}
        </div>
        <div className={`pulse-dot ${running ? "on" : ""}`} title={running ? "Run in progress" : "Idle"} />
      </nav>
      <div className="main">
        <div className="brandbar">
          <div className="brand">AUTO<span>PLANT</span></div>
          <TopBar route={route} />
        </div>
        <AnimatePresence>
          {error && (
            <motion.div className="banner" initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }}>{error}</motion.div>
          )}
        </AnimatePresence>
        <main className="stage">
          <AnimatePresence mode="wait"><Page key={route} /></AnimatePresence>
        </main>
      </div>
      <MemoryDrawer />
      <TraceModal />
    </div>
  );
}
