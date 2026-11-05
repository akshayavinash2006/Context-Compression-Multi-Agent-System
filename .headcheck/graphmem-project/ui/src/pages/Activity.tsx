import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Card, Empty, Page } from "../components/ui";
import { zoneFor } from "../layout";
import { useStore } from "../store";

const FILTERS = [["all", "All"], ["write", "Writes"], ["read", "Reads"], ["llm", "Model calls"], ["answer", "Answers"], ["question", "Questions"]] as const;

export function Activity() {
  const { feed, set } = useStore();
  const [filter, setFilter] = useState<string>("all");
  const rows = feed.filter((f) => filter === "all" || f.kind === filter);
  return (
    <Page title="Activity" sub="Every read, write and model call, newest first."
      actions={<div className="seg">{FILTERS.map(([id, l]) => (
        <button key={id} className={filter === id ? "on" : ""} onClick={() => setFilter(id)}>
          {filter === id && <motion.i layoutId="seg-pill" transition={{ type: "spring", stiffness: 420, damping: 34 }} />}<span>{l}</span>
        </button>))}</div>}>
      <Card pad={false} className="list-card">
        {rows.length === 0 ? <Empty>{feed.length ? "Nothing matches this filter." : "Start a run to see agents write and read shared memory."}</Empty> : (
          <ul className="timeline">
            <AnimatePresence initial={false}>
              {rows.slice(0, 120).map((f) => (
                <motion.li key={f.id} layout="position" initial={{ opacity: 0, x: -14 }} animate={{ opacity: 1, x: 0 }} className={f.memoryId ? "click" : ""} onClick={() => f.memoryId && set({ openMemory: f.memoryId })}>
                  <span className="t-turn">t{f.turn}</span>
                  <span className="t-dot" style={{ background: f.tagColor, boxShadow: `0 0 10px ${f.tagColor}` }} />
                  <div className="t-body">
                    <div className="t-meta">
                      <span className="ftag" style={{ color: f.tagColor }}>{f.tag}</span>
                      {f.agent && <span style={{ color: zoneFor(f.agent).color }}>{zoneFor(f.agent).name}</span>}
                      {f.memoryId && <code className="chip">{f.memoryId}</code>}
                    </div>
                    <p>{f.text}</p>
                  </div>
                </motion.li>
              ))}
            </AnimatePresence>
          </ul>
        )}
      </Card>
    </Page>
  );
}
