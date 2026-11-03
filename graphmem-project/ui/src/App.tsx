import { useEffect } from "react";
import { connect, loadInitial, startRun } from "./api";
import { AgentPanel } from "./panels/AgentPanel";
import { Analytics } from "./panels/Analytics";
import { Feed } from "./panels/Feed";
import { MemoryDrawer, TraceModal } from "./panels/Overlays";
import { Questions } from "./panels/Questions";
import { TopBar } from "./panels/TopBar";
import { Factory } from "./scene/Factory";
import { settleAgents, useStore } from "./store";

let autostarted = false;

export default function App() {
  const error = useStore((s) => s.error);
  useEffect(() => {
    connect();
    loadInitial();
    const id = window.setInterval(settleAgents, 300);
    return () => window.clearInterval(id);
  }, []);

  // Demo link: /?autostart=P001&preset=mock&delay=0.5 starts a run as soon as the server is connected.
  const wsStatus = useStore((s) => s.wsStatus);
  useEffect(() => {
    const q = new URLSearchParams(location.search);
    const sc = q.get("autostart");
    if (sc && wsStatus === "connected" && !autostarted) {
      autostarted = true;
      startRun({ scenario: sc, preset: q.get("preset") ?? "mock", condition: q.get("condition") ?? "graph",
        mode: q.get("mode") ?? "replay", delay: Number(q.get("delay") ?? 0.5) });
    }
  }, [wsStatus]);

  return (
    <div className="app">
      <nav className="sidebar" aria-label="Main">
        <div className="logo" title="AutoPlant GraphMem">◆</div>
        <button className="nav-item active" title="Plant floor">
          <svg viewBox="0 0 24 24"><path d="M3 21V10l6 4V10l6 4V6l6 4v11z" /></svg>
        </button>
      </nav>
      <div className="main">
        <TopBar />
        {error && <div className="banner">{error}</div>}
        <div className="content">
          <section className="card scene">
            <Factory />
            <div className="scene-legend">
              <span><i style={{ background: "#22d3ee" }} />writing</span>
              <span><i style={{ background: "#fbbf24" }} />reading</span>
              <span><i style={{ background: "#a78bfa" }} />calling LLM</span>
              <span><i style={{ background: "#4ade80" }} />duplicate merged</span>
              <span><i style={{ background: "#f87171" }} />fact replaced</span>
            </div>
          </section>
          <div className="right"><Analytics /><AgentPanel /></div>
          <div className="bottom"><Feed /><Questions /></div>
        </div>
      </div>
      <MemoryDrawer />
      <TraceModal />
    </div>
  );
}
