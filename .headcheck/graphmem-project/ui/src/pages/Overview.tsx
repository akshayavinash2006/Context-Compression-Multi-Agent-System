import { AnimatePresence, motion } from "framer-motion";
import { AgentSheet } from "../components/AgentSheet";
import { Counter, Spark } from "../components/ui";
import { ZONES } from "../layout";
import { go } from "../router";
import { Factory } from "../scene/Factory";
import { STATUS_COLOR, STATUS_LABEL } from "../status";
import { useStore } from "../store";

const LEGEND = [["#2ee6d0", "Writing"], ["#fbbf24", "Reading"], ["#a78bfa", "Calling LLM"], ["#4ade80", "Merged"], ["#f87171", "Replaced"]];

function SceneOverlay() {
  const { cfg, scenarios, turn, totalEvents, question, wsStatus } = useStore();
  const sc = scenarios.find((s) => s.id === cfg.scenario);
  return (
    <>
      <div className="scene-title">
        <span className="eyebrow">{wsStatus === "running" ? "Live shift" : "Plant floor"}</span>
        <h1>{sc ? `Scenario ${sc.id}` : "AutoPlant"}</h1>
        <p>{sc ? `${sc.events} events · ${sc.questions} diagnosis questions` : "Generate a scenario in run settings, then start a run."}</p>
        {totalEvents > 0 && <p className="turn">Event <b><Counter value={turn} /></b> of {totalEvents}</p>}
      </div>
      <div className="scene-legend">{LEGEND.map(([c, l]) => <span key={l}><i style={{ background: c, boxShadow: `0 0 8px ${c}` }} />{l}</span>)}</div>
      <AnimatePresence>
        {question && (
          <motion.div className="scene-question" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 16 }}>
            <span className="eyebrow">Now diagnosing</span>
            <p>{question.text}</p>
            <div className="q-bar"><span /></div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

function Rail() {
  const { tokens, graphCum, fullCum, condition, agents, counts, feed, selectedAgent, set } = useStore();
  const saved = fullCum > 0 ? Math.round((1 - graphCum / fullCum) * 100) : 0;
  return (
    <aside className="rail">
      <AnimatePresence mode="wait">
        {selectedAgent ? (
          <AgentSheet key="sheet" id={selectedAgent} onClose={() => set({ selectedAgent: undefined })} />
        ) : (
          <motion.div key="rail" className="rail-stack" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <section className="card glass hero-kpi">
              <span className="eyebrow">Tokens sent to models</span>
              <div className="kpi"><b><Counter value={graphCum} /></b>{fullCum > 0 && condition !== "full" && <em className="delta up">−{saved}%</em>}</div>
              <p className="muted">{fullCum > 0 && condition !== "full" ? <>vs <Counter value={fullCum} /> with full history</> : "Starts counting with the first model call"}</p>
              <Spark values={tokens.map((t) => t.graph)} />
              <button className="link-btn" onClick={() => go("analytics")}>Open analytics →</button>
            </section>

            <section className="card glass">
              <header className="card-head"><h2>Agents</h2><span className="muted">{counts.active} memories</span></header>
              <ul className="roster">
                {ZONES.map((z) => {
                  const st = agents[z.id]?.status ?? "idle";
                  return (
                    <li key={z.id} onClick={() => set({ selectedAgent: z.id })}>
                      <span className="r-dot" style={{ background: STATUS_COLOR[st], boxShadow: st !== "idle" ? `0 0 10px ${STATUS_COLOR[st]}` : "none" }} />
                      <span className="r-name">{z.name}</span>
                      <span className="r-status" style={{ color: st === "idle" ? undefined : STATUS_COLOR[st] }}>{st === "idle" ? "" : STATUS_LABEL[st]}</span>
                    </li>
                  );
                })}
              </ul>
            </section>

            <section className="card glass">
              <header className="card-head"><h2>Latest</h2><button className="link-btn" onClick={() => go("activity")}>All activity →</button></header>
              <ul className="ticker">
                <AnimatePresence initial={false}>
                  {feed.slice(0, 4).map((f) => (
                    <motion.li key={f.id} initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                      <span className="ftag" style={{ color: f.tagColor }}>{f.tag}</span>
                      <span className="ttext">{f.text}</span>
                    </motion.li>
                  ))}
                </AnimatePresence>
                {feed.length === 0 && <li className="muted">Waiting for the first event…</li>}
              </ul>
            </section>
          </motion.div>
        )}
      </AnimatePresence>
    </aside>
  );
}

export function Overview() {
  return (
    <motion.div className="overview" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0, transition: { duration: 0.15 } }}>
      <motion.div className="scene" initial={{ opacity: 0, scale: 0.985 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.6, ease: "easeOut" }}>
        <Factory />
        <SceneOverlay />
      </motion.div>
      <Rail />
    </motion.div>
  );
}
