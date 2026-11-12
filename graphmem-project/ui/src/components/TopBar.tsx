import { motion } from "framer-motion";
import { stopRun } from "../api";
import { ROUTES, go, type Route } from "../router";
import { useStore } from "../store";

const STATUS_TEXT: Record<string, string> = {
  disconnected: "Server offline", connected: "Ready", running: "Live", done: "Finished", stopped: "Stopped", error: "Error",
};

export function TopBar({ route }: { route: Route }) {
  const { wsStatus, starting, turn, totalEvents } = useStore();
  const running = wsStatus === "running";
  const pct = totalEvents ? Math.round((turn / totalEvents) * 100) : 0;
  const meta = ROUTES.find((r) => r.id === route)!;
  const label = starting ? "Preparing" : STATUS_TEXT[wsStatus];
  const cls = starting ? "starting" : wsStatus;

  return (
    <header className="topbar">
      <div className="crumb">
        <span className="crumb-title">{meta.label}</span>
        <span className="crumb-hint">{meta.hint}</span>
      </div>
      <div className="topbar-right">
        <div className={`status ${cls}`} role="status">
          <i />{label}
          {totalEvents > 0 && !starting && <em>{turn}/{totalEvents}</em>}
        </div>
        {(running || starting) && (
          <div className="progress" aria-label={starting ? "Preparing the run" : `${pct}% of events processed`}>
            {starting ? <span className="indeterminate" /> : <motion.span animate={{ width: `${pct}%` }} transition={{ ease: "easeOut", duration: 0.4 }} />}
          </div>
        )}
        {running
          ? <button className="btn danger" onClick={stopRun}>Stop</button>
          : route === "run"
            ? null
            : <motion.button className="btn primary" whileTap={{ scale: 0.97 }} onClick={() => go("run")}>
                <svg viewBox="0 0 24 24"><path d="M7 5v14l12-7z" /></svg>{starting ? "View progress" : "New run"}
              </motion.button>}
      </div>
    </header>
  );
}
