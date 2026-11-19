import { useEffect, useState, type ReactNode } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { generateScenarios, getJSON, startRun, stopRun } from "../api";
import { Counter, Page, rise, stagger } from "../components/ui";
import { go } from "../router";
import { useStore } from "../store";

interface Choice { id: string; title: string; tag?: string; text: string; needs?: string }

const PRESETS: Choice[] = [
  { id: "mock", title: "Mock", tag: "Fastest", text: "A built-in rule-based stand-in for a language model. Answers are deterministic and instant, so it is ideal for trying the interface and checking the memory logic.", needs: "Nothing. No GPU, no network." },
  { id: "single_provider", title: "Single provider", tag: "Controlled", text: "Every agent runs on the same local model (Ollama, Qwen 2.5 7B). Use it for fair comparisons, because the only thing that changes between runs is the memory.", needs: "Ollama installed with the qwen2.5 models pulled." },
  { id: "multi_provider", title: "Multi provider", tag: "Realistic", text: "Each agent gets the model that suits its job: small local models for the sensors, a stronger hosted model for diagnosis and reports.", needs: "Ollama plus Groq and Google API keys in the server's .env." },
];

const MEMORY: Choice[] = [
  { id: "graph", title: "Graph memory", tag: "The system", text: "Agents share one graph. Every message is compressed, deduplicated and linked, and each agent receives only a small ranked slice of it." },
  { id: "full", title: "Full history", tag: "Baseline", text: "Every agent re-reads the whole conversation so far. Most accurate in principle, but the token cost grows quickly." },
  { id: "vector", title: "Vector RAG", tag: "Baseline", text: "Messages are stored as embeddings and the closest matches to the question are retrieved." },
  { id: "window", title: "Last 10 messages", tag: "Baseline", text: "Agents only see the ten most recent messages. Cheap, but older facts are forgotten." },
  { id: "summary", title: "Rolling summary", tag: "Baseline", text: "A running summary replaces older messages. Compact, but details can be lost in the summary." },
];

const MODES: Choice[] = [
  { id: "replay", title: "Replay", tag: "Cheaper", text: "Sensor and maintenance messages are replayed from the scenario as written. Only the Diagnosis and Report agents call a language model." },
  { id: "live", title: "Live", tag: "Slower", text: "Every agent generates its own messages with a language model, so the whole pipeline is exercised. Needs a real model preset to be meaningful." },
];

interface ModelsInfo { roles: Record<string, { primary: string; fallbacks: string[]; skipped: string[] }> }

function Section({ n, title, intro, children }: { n: number; title: string; intro: string; children: ReactNode }) {
  return (
    <motion.section variants={rise} className="card cfg-section">
      <header className="cfg-head"><span className="step-no">{n}</span><div><h2>{title}</h2><p>{intro}</p></div></header>
      {children}
    </motion.section>
  );
}

function Choices({ items, value, onPick, disabled, cols = 3 }: { items: Choice[]; value: string; onPick: (id: string) => void; disabled: boolean; cols?: number }) {
  return (
    <div className="choices" style={{ ["--cols" as string]: cols }}>
      {items.map((c) => (
        <button key={c.id} type="button" disabled={disabled} aria-pressed={value === c.id} className={`choice ${value === c.id ? "on" : ""}`} onClick={() => onPick(c.id)}>
          {value === c.id && <motion.i layoutId={`pick-${items[0].id}`} className="choice-ring" transition={{ type: "spring", stiffness: 420, damping: 34 }} />}
          <span className="choice-top"><b>{c.title}</b>{c.tag && <em>{c.tag}</em>}</span>
          <span className="choice-text">{c.text}</span>
          {c.needs && <span className="choice-needs">Needs: {c.needs}</span>}
        </button>
      ))}
    </div>
  );
}

function ModelLineup({ preset }: { preset: string }) {
  const [info, setInfo] = useState<ModelsInfo>();
  const [err, setErr] = useState<string>();
  useEffect(() => {
    setInfo(undefined); setErr(undefined);
    let dead = false;
    getJSON<ModelsInfo>(`/api/models?preset=${preset}`).then((d) => !dead && setInfo(d)).catch((e) => !dead && setErr(String(e.message ?? e)));
    return () => { dead = true; };
  }, [preset]);
  if (err) return <p className="note bad">Could not check the models: {err}</p>;
  if (!info) return <p className="note"><span className="spinner" /> Checking which models are available…</p>;
  const roles = Object.entries(info.roles).filter(([r]) => r !== "summariser");
  const problems = roles.flatMap(([r, v]) => v.skipped.map((s) => `${r}: ${s}`));
  return (
    <div className="lineup">
      <div className="label">Models this preset will use</div>
      <ul>
        {roles.map(([role, v]) => (
          <li key={role}><span>{role}</span><code>{v.primary}</code>{v.fallbacks.length > 0 && <em>fallback {v.fallbacks[0]}</em>}</li>
        ))}
      </ul>
      {problems.length > 0 && <p className="note warn">Unavailable and skipped: {problems.join("; ")}. The run falls back along the chain, and fails if no model works.</p>}
    </div>
  );
}

function Slider({ value, onChange, disabled, events }: { value: number; onChange: (v: number) => void; disabled: boolean; events: number }) {
  const est = events * value;
  return (
    <div className="slider">
      <div className="slider-top">
        <b>{value === 0 ? "No pause" : `${value.toFixed(1)} s per event`}</b>
        <span>{events ? (est < 1 ? "finishes almost instantly" : `about ${Math.round(est)} s of pacing for ${events} events`) : "pick a scenario to see the estimate"}</span>
      </div>
      <input type="range" min={0} max={2} step={0.1} value={value} disabled={disabled} onChange={(e) => onChange(+e.target.value)} aria-label="Pace in seconds per event" style={{ ["--p" as string]: `${(value / 2) * 100}%` }} />
      <div className="slider-scale"><span>Instant</span><span>Watchable</span><span>Slow</span></div>
    </div>
  );
}

const fmt = (ms: number) => { const s = Math.max(0, Math.floor(ms / 1000)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };

function Progress() {
  const { wsStatus, starting, turn, totalEvents, startedAt, finishedAt, graphCum, counts, answers, question, error } = useStore();
  const [tick, setTick] = useState(0);
  const live = starting || wsStatus === "running";
  useEffect(() => {
    if (!live) return;
    const id = window.setInterval(() => setTick((t) => t + 1), 250);
    return () => window.clearInterval(id);
  }, [live]);
  void tick;
  const finished = !starting && ["done", "stopped", "error"].includes(wsStatus);
  const pct = totalEvents ? Math.round((turn / totalEvents) * 100) : 0;
  const elapsed = startedAt ? (finishedAt ?? performance.now()) - startedAt : 0;
  const correct = answers.filter((a) => a.correct).length;
  const steps = [
    { id: "connect", label: "Connected to server", state: wsStatus === "disconnected" ? "wait" : "done" },
    { id: "prepare", label: "Preparing models and memory", state: starting ? "active" : startedAt || finished ? "done" : "wait" },
    { id: "events", label: "Processing plant events", state: starting ? "wait" : wsStatus === "running" ? "active" : finished && turn > 0 ? "done" : "wait" },
    { id: "done", label: wsStatus === "stopped" ? "Stopped" : wsStatus === "error" ? "Failed" : "Finished", state: finished ? (wsStatus === "done" ? "done" : "bad") : "wait" },
  ];
  const idle = !live && !finished;

  return (
    <div className="progress-panel">
      {idle && <p className="muted">Nothing is running. Choose your options on the left and press <b>Start run</b>. Progress appears here as soon as you do.</p>}
      {(live || finished) && (
        <>
          <ol className="steps">
            {steps.map((s) => (
              <li key={s.id} className={s.state}><i>{s.state === "done" ? "✓" : s.state === "bad" ? "!" : ""}</i><span>{s.label}</span></li>
            ))}
          </ol>
          {starting && <p className="note"><span className="spinner" /> Getting everything ready. The first run after the server starts can take 10 to 30 seconds while models load. {fmt(elapsed)} elapsed.</p>}
          {!starting && (
            <>
              <div className="bigbar" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
                <motion.span animate={{ width: `${pct}%` }} transition={{ ease: "easeOut", duration: 0.4 }} className={wsStatus === "running" ? "live" : ""} />
              </div>
              <div className="bigbar-meta"><b><Counter value={turn} /> / {totalEvents} events</b><span>{pct}% · {fmt(elapsed)}</span></div>
              <div className="run-stats">
                <div><b><Counter value={graphCum} /></b><span>tokens sent</span></div>
                <div><b><Counter value={counts.active} /></b><span>memories</span></div>
                <div><b>{answers.length ? `${correct}/${answers.length}` : "–"}</b><span>correct answers</span></div>
              </div>
            </>
          )}
          <AnimatePresence>
            {question && wsStatus === "running" && (
              <motion.div className="now-q" initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }}>
                <span className="eyebrow live">Diagnosing</span><p>{question.text}</p>
              </motion.div>
            )}
          </AnimatePresence>
          {error && <p className="note bad">{error}</p>}
          {finished && (
            <div className="after">
              <button className="btn primary" onClick={() => go("overview")}>Watch the plant</button>
              <button className="btn" onClick={() => go("analytics")}>Memory analytics</button>
              <button className="btn" onClick={() => go("diagnosis")}>Answers</button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export function Run() {
  const { cfg, setCfg, scenarios, wsStatus, starting } = useStore();
  const plant = scenarios.filter((s) => s.has_zones);
  const sc = plant.find((s) => s.id === cfg.scenario);
  const running = wsStatus === "running" || starting;
  const offline = wsStatus === "disconnected";
  const [generating, setGenerating] = useState(false);

  useEffect(() => { if (!cfg.scenario && plant.length) setCfg({ scenario: plant[0].id }); }, [plant, cfg.scenario, setCfg]);

  const canStart = !offline && !!cfg.scenario && !running;
  const why = offline ? "The backend is not reachable. Start the server on port 8000." : !cfg.scenario ? "Choose a scenario first." : "";
  const preset = PRESETS.find((p) => p.id === cfg.preset)?.title ?? cfg.preset;
  const mem = MEMORY.find((m) => m.id === cfg.condition)?.title ?? cfg.condition;

  return (
    <Page title="Run" sub="Choose how the plant is simulated, press Start, and follow the progress here.">
      <div className="run-layout">
        <motion.div className="cfg-col" variants={stagger}>
          <Section n={1} title="Scenario" intro="A scenario is one simulated shift: a stream of sensor readings and maintenance notes, with diagnosis questions along the way. Each has a known right answer, so the run can be scored.">
            <div className="scenario-row">
              <label className="field">
                <span>Scenario</span>
                <select value={cfg.scenario} disabled={running} onChange={(e) => setCfg({ scenario: e.target.value })}>
                  {plant.length === 0 && <option value="">No scenarios yet</option>}
                  {plant.map((s) => <option key={s.id} value={s.id}>{s.id} · {s.events} events · {s.questions} questions</option>)}
                </select>
              </label>
              <button className="btn ghost" disabled={running || generating} onClick={async () => { setGenerating(true); await generateScenarios(10); setGenerating(false); }}>
                {generating ? "Generating…" : "Generate 10 new"}
              </button>
            </div>
            {sc && (
              <div className="facts">
                <span><b>{sc.events}</b> events</span><span><b>{sc.questions}</b> questions</span><span><b>{sc.machines.length}</b> machines</span>
                {sc.injected && <span className="warn-chip">contains an injected wrong record</span>}
              </div>
            )}
          </Section>

          <Section n={2} title="Models" intro="Which language models power the agents. The preset decides which model each agent uses.">
            <Choices items={PRESETS} value={cfg.preset} onPick={(id) => setCfg({ preset: id })} disabled={running} />
            <ModelLineup preset={cfg.preset} />
          </Section>

          <Section n={3} title="Memory" intro="How agents remember what happened earlier. Graph memory is the system being tested; the others are baselines to compare it against on tokens used and answer accuracy.">
            <Choices items={MEMORY} value={cfg.condition} onPick={(id) => setCfg({ condition: id })} disabled={running} />
          </Section>

          <Section n={4} title="Mode" intro="How much of the pipeline uses a language model.">
            <Choices items={MODES} value={cfg.mode} onPick={(id) => setCfg({ mode: id })} disabled={running} cols={2} />
          </Section>

          <Section n={5} title="Pace" intro="A pause after each event so you can watch the plant react. It only slows the display, never the answers. Set it to zero for the quickest result.">
            <Slider value={cfg.delay} onChange={(v) => setCfg({ delay: v })} disabled={running} events={sc?.events ?? 0} />
          </Section>
        </motion.div>

        <motion.aside className="run-col" variants={rise}>
          <div className="card run-card">
            <header className="card-head"><h2>Your run</h2></header>
            <div className="card-body">
              <div className="summary-chips">
                <span>{sc ? sc.id : "No scenario"}</span><span>{preset}</span><span>{mem}</span><span>{cfg.mode === "live" ? "Live" : "Replay"}</span><span>{cfg.delay === 0 ? "No pause" : `${cfg.delay.toFixed(1)}s pace`}</span>
              </div>
              {wsStatus === "running"
                ? <button className="btn danger big" onClick={stopRun}>Stop run</button>
                : <motion.button className={`btn primary big ${starting ? "busy" : ""}`} disabled={!canStart} whileTap={{ scale: 0.98 }} onClick={() => startRun(cfg)}>
                    {starting ? <><span className="spinner dark" />Starting…</> : <><svg viewBox="0 0 24 24"><path d="M7 5v14l12-7z" /></svg>Start run</>}
                  </motion.button>}
              {why && !running && <p className="note warn">{why}</p>}
              <Progress />
            </div>
          </div>
        </motion.aside>
      </div>
    </Page>
  );
}
