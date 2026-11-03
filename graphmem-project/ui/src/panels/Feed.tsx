import { zoneFor } from "../layout";
import { useStore } from "../store";

export function Feed() {
  const { feed, set } = useStore();
  return (
    <section className="card feed">
      <div className="card-head"><h2>Live Feed</h2><span className="muted-text">{feed.length} events</span></div>
      <ul className="feed-list">
        {feed.length === 0 && <li className="empty">Start a run to see agents write and read shared memory.</li>}
        {feed.map((f) => (
          <li key={f.id} className={f.memoryId ? "clickable" : ""} onClick={() => f.memoryId && set({ openMemory: f.memoryId })}>
            <span className="turn-no">t{f.turn}</span>
            {f.tag && <span className="ftag" style={{ color: f.tagColor, borderColor: f.tagColor }}>{f.tag}</span>}
            {f.agent && <span className="fagent" style={{ color: zoneFor(f.agent).color }}>{zoneFor(f.agent).name}</span>}
            <span className="ftext">{f.memoryId && <code>{f.memoryId}</code>} {f.text}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
