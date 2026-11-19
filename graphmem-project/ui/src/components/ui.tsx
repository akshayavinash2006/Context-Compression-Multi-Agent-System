import { useEffect, useRef, useState, type ReactNode } from "react";
import { motion } from "framer-motion";

export function useCountUp(value: number, ms = 800) {
  const [v, setV] = useState(value);
  const from = useRef(value);
  useEffect(() => {
    const a = from.current;
    const start = performance.now();
    let raf = 0;
    const tick = (t: number) => {
      const k = Math.min(1, (t - start) / ms);
      const cur = a + (value - a) * (1 - Math.pow(1 - k, 3));
      from.current = cur;
      setV(cur);
      if (k < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value, ms]);
  return v;
}

export function Counter({ value, decimals = 0, suffix = "" }: { value: number; decimals?: number; suffix?: string }) {
  const raw = useCountUp(value);
  const v = Math.abs(raw) < 0.5 / 10 ** decimals ? 0 : raw;   // avoids a flash of "-0"
  return <>{v.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}{suffix}</>;
}

export const stagger = { hidden: {}, show: { transition: { staggerChildren: 0.06 } } };
export const rise = { hidden: { opacity: 0, y: 14 }, show: { opacity: 1, y: 0, transition: { type: "spring" as const, stiffness: 260, damping: 26 } } };

export function Card({ title, aside, children, className = "", pad = true }: { title?: ReactNode; aside?: ReactNode; children: ReactNode; className?: string; pad?: boolean }) {
  return (
    <motion.section variants={rise} className={`card ${className}`} whileHover={{ y: -2 }} transition={{ duration: 0.2 }}>
      {title && <header className="card-head"><h2>{title}</h2>{aside}</header>}
      <div className={pad ? "card-body" : ""}>{children}</div>
    </motion.section>
  );
}

export function Page({ title, sub, actions, children }: { title: string; sub?: string; actions?: ReactNode; children: ReactNode }) {
  return (
    <motion.div className="page" variants={stagger} initial="hidden" animate="show" exit={{ opacity: 0, y: -8, transition: { duration: 0.15 } }}>
      <motion.div variants={rise} className="page-head">
        <div><h1>{title}</h1>{sub && <p>{sub}</p>}</div>
        {actions}
      </motion.div>
      {children}
    </motion.div>
  );
}

export function Stat({ label, value, decimals = 0, suffix, tone, note }: { label: string; value: number | string; decimals?: number; suffix?: string; tone?: string; note?: string }) {
  return (
    <motion.div variants={rise} className="stat">
      <span className="stat-label">{label}</span>
      <b className="stat-value" style={tone ? { color: tone } : undefined}>
        {typeof value === "number" ? <Counter value={value} decimals={decimals} suffix={suffix} /> : value}
      </b>
      {note && <span className="stat-note">{note}</span>}
    </motion.div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty"><span className="empty-orb" />{children}</div>;
}

// Donut with animated segments.
export function Donut({ parts, size = 150, center, sub }: { parts: { label: string; value: number; color: string }[]; size?: number; center?: ReactNode; sub?: string }) {
  const total = parts.reduce((a, p) => a + p.value, 0);
  const r = 52, C = 2 * Math.PI * r;
  let acc = 0;
  return (
    <div className="donut" style={{ width: size, height: size }}>
      <svg viewBox="0 0 140 140">
        <circle cx="70" cy="70" r={r} className="donut-track" />
        {total > 0 && parts.map((p) => {
          const len = (p.value / total) * C;
          const el = <circle key={p.label} cx="70" cy="70" r={r} className="donut-seg" stroke={p.color}
            strokeDasharray={`${Math.max(0, len - 3)} ${C}`} strokeDashoffset={-acc} />;
          acc += len;
          return el;
        })}
      </svg>
      <div className="donut-center"><b>{center}</b>{sub && <span>{sub}</span>}</div>
    </div>
  );
}

export function Gauge({ value, label, color = "var(--accent)" }: { value: number; label: string; color?: string }) {
  const v = Math.max(0, Math.min(1, value));
  const r = 46, C = Math.PI * r;
  return (
    <div className="gauge">
      <svg viewBox="0 0 120 70">
        <path d="M14 64 A46 46 0 0 1 106 64" className="gauge-track" />
        <path d="M14 64 A46 46 0 0 1 106 64" className="gauge-fill" style={{ stroke: color, strokeDasharray: `${C * v} ${C}` }} />
      </svg>
      <div className="gauge-num"><Counter value={v * 100} suffix="%" /></div>
      <span className="gauge-label">{label}</span>
    </div>
  );
}

export function Spark({ values, color = "var(--accent)", h = 44 }: { values: number[]; color?: string; h?: number }) {
  const W = 200;
  if (values.length < 2) return <div className="spark-empty" style={{ height: h }} />;
  const max = Math.max(...values, 1);
  const pts = values.map((v, i) => [(i / (values.length - 1)) * W, h - 3 - (v / max) * (h - 8)] as const);
  const d = pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const last = pts[pts.length - 1];
  return (
    <svg viewBox={`0 0 ${W} ${h}`} className="spark" preserveAspectRatio="none" style={{ height: h }}>
      <defs>
        <linearGradient id="sp" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor={color} stopOpacity=".35" /><stop offset="1" stopColor={color} stopOpacity="0" /></linearGradient>
      </defs>
      <path d={`${d} L${W},${h} L0,${h} Z`} fill="url(#sp)" />
      <path d={d} fill="none" stroke={color} strokeWidth="2" vectorEffect="non-scaling-stroke" />
      <circle cx={last[0]} cy={last[1]} r="3.2" fill={color} className="spark-dot" />
    </svg>
  );
}
