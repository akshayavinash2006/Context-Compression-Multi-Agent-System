import { useStore } from "../store";

const label = (s: string) => s.replace(/_/g, " ");

export function Questions() {
  const { question, answers, set } = useStore();
  return (
    <section className="card questions">
      <div className="card-head"><h2>Diagnosis Tracker</h2>
        <span className="muted-text">{answers.filter((a) => a.correct).length}/{answers.length} correct</span></div>
      {question && (
        <div className="q-current">
          <div className="q-label">Now diagnosing</div>
          <div className="q-text">{question.text}</div>
          <div className="q-progress"><span /></div>
        </div>
      )}
      <ul className="q-list">
        {answers.length === 0 && !question && <li className="empty">Questions appear here as the shift progresses.</li>}
        {[...answers].reverse().map((a) => (
          <li key={a.event_id} onClick={() => set({ openTrace: a.report_memory_id })} title="Show the trace of this answer">
            <span className={`verdict ${a.correct ? "ok" : "bad"}`}>{a.correct ? "✓" : "✗"}</span>
            <div className="q-body">
              <div className="q-text">{a.question}</div>
              <div className="q-meta">
                <b>{a.report.machine || "?"}</b> · {label(a.report.root_cause)}
                {!a.correct && <span className="expected"> (expected {a.gold.machine} · {label(a.gold.root_cause)})</span>}
              </div>
              <div className="q-action">{a.report.action}</div>
            </div>
            <span className="trace-link">Trace →</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
