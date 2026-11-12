import { label, type ParsedReport } from "../report";
import { useStore } from "../store";

interface Props {
  report: ParsedReport;
  question?: string;
  verdict?: boolean;              // true correct, false incorrect, undefined unknown
  expected?: { machine: string; root_cause: string };
  traceId?: string;               // memory id of the report, opens its trace
}

// A maintenance report laid out as labelled fields instead of raw JSON.
export function ReportCard({ report, question, verdict, expected, traceId }: Props) {
  const set = useStore((s) => s.set);
  return (
    <article className={`rcard ${verdict === undefined ? "" : verdict ? "ok" : "bad"}`}>
      {(question || verdict !== undefined) && (
        <header>
          {verdict !== undefined && <span className="verdict">{verdict ? "Correct" : "Incorrect"}</span>}
          {question && <p className="rq">{question}</p>}
        </header>
      )}
      <dl>
        <dt>Machine</dt><dd><b>{report.machine || "Not identified"}</b></dd>
        <dt>Root cause</dt><dd className="cause">{label(report.root_cause)}</dd>
        <dt>Action</dt><dd>{report.action || "No action recommended"}</dd>
        <dt>Evidence</dt>
        <dd className="chips">
          {report.evidence.length === 0 ? <span className="muted">None cited</span>
            : report.evidence.map((e) => <code key={e} className="chip">{e}</code>)}
        </dd>
      </dl>
      {verdict === false && expected && <p className="rexp">Expected {expected.machine} · {label(expected.root_cause)}</p>}
      {traceId && (
        <footer>
          <button className="link-btn" onClick={() => set({ openMemory: traceId })}>Open memory</button>
          <button className="link-btn" onClick={() => set({ openTrace: traceId })}>Show trace →</button>
        </footer>
      )}
    </article>
  );
}
