import { AnimatePresence, motion } from "framer-motion";
import { Card, Empty, Gauge, Page, rise } from "../components/ui";
import { useStore } from "../store";

const label = (s: string) => s.replace(/_/g, " ");

export function Diagnosis() {
  const { question, answers, set } = useStore();
  const correct = answers.filter((a) => a.correct).length;
  return (
    <Page title="Diagnosis" sub="Each question the plant asks, the answer the agents gave, and the facts behind it.">
      <div className="grid-main">
        <Card className="q-hero">
          <AnimatePresence mode="wait">
            {question ? (
              <motion.div key={question.eventId} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -12 }}>
                <span className="eyebrow live">Now diagnosing</span>
                <p className="q-big">{question.text}</p>
                <div className="q-bar"><span /></div>
              </motion.div>
            ) : (
              <motion.div key="idle" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                <span className="eyebrow">Standing by</span>
                <p className="q-big muted">Questions appear here as the shift progresses.</p>
              </motion.div>
            )}
          </AnimatePresence>
        </Card>
        <Card title="Accuracy">
          <div className="center-col"><Gauge value={answers.length ? correct / answers.length : 0} label={answers.length ? `${correct} of ${answers.length} correct` : "No answers yet"} color="var(--ok)" /></div>
        </Card>
      </div>

      {answers.length === 0 ? <Card><Empty>No answers yet.</Empty></Card> : (
        <motion.div className="answers" layout>
          <AnimatePresence initial={false}>
            {[...answers].reverse().map((a) => (
              <motion.article key={a.event_id} layout variants={rise} initial="hidden" animate="show" className={`answer ${a.correct ? "ok" : "bad"}`} whileHover={{ y: -3 }} onClick={() => set({ openTrace: a.report_memory_id })}>
                <div className="a-top">
                  <span className="verdict">{a.correct ? "Correct" : "Incorrect"}</span>
                  <span className="muted">Open trace →</span>
                </div>
                <p className="a-q">{a.question}</p>
                <div className="a-main"><b>{a.report.machine || "Unknown machine"}</b><span>{label(a.report.root_cause)}</span></div>
                <p className="a-action">{a.report.action}</p>
                {!a.correct && <p className="a-exp">Expected {a.gold.machine} · {label(a.gold.root_cause)}</p>}
              </motion.article>
            ))}
          </AnimatePresence>
        </motion.div>
      )}
    </Page>
  );
}
