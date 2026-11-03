import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import { Color, Mesh, MeshBasicMaterial } from "three";
import type { ZoneLayout } from "../layout";
import { useStore, type AgentStatus } from "../store";
import { Model } from "./Model";

export const STATUS_COLOR: Record<AgentStatus, string> = {
  idle: "#334155", reading: "#fbbf24", thinking: "#a78bfa", writing: "#22d3ee",
};
const STATUS_LABEL: Record<AgentStatus, string> = {
  idle: "Idle", reading: "Reading memory", thinking: "Calling LLM", writing: "Writing to memory",
};

export function Zone({ zone }: { zone: ZoneLayout }) {
  const agent = useStore((s) => s.agents[zone.id]);
  const selected = useStore((s) => s.selectedAgent === zone.id);
  const inRun = useStore((s) => s.zonesInRun.length === 0 || zone.kind === "support" || s.zonesInRun.includes(zone.id));
  const set = useStore((s) => s.set);
  const ring = useRef<Mesh>(null);
  const status: AgentStatus = agent?.status ?? "idle";
  const target = new Color(status === "idle" ? zone.color : STATUS_COLOR[status]);

  // Ring glows in the zone colour when idle and pulses in the status colour when the agent is busy.
  useFrame(({ clock }) => {
    const mesh = ring.current;
    if (!mesh) return;
    const m = mesh.material as MeshBasicMaterial;
    const active = status !== "idle";
    const t = clock.elapsedTime;
    m.color.copy(target).multiplyScalar(active ? 1.9 + Math.sin(t * 8) * 0.7 : 0.6);
    m.opacity = active ? 0.95 : 0.5;
    mesh.scale.setScalar(active ? 1 + Math.sin(t * 6) * 0.025 : 1);
  });

  const [x, z] = zone.pos;
  return (
    <group position={[x, 0, z]}>
      {/* pad */}
      <mesh rotation-x={-Math.PI / 2} position={[0, 0.02, 0]} receiveShadow
        onClick={(e) => { e.stopPropagation(); set({ selectedAgent: zone.id }); }}>
        <circleGeometry args={[5.4, 48]} />
        <meshStandardMaterial color={selected ? "#16243a" : "#0f1a2b"} />
      </mesh>
      {/* status ring */}
      <mesh ref={ring} rotation-x={-Math.PI / 2} position={[0, 0.05, 0]}>
        <ringGeometry args={[5.2, 5.5, 64]} />
        <meshBasicMaterial transparent toneMapped={false} color={zone.color} />
      </mesh>
      <group onClick={(e) => { e.stopPropagation(); set({ selectedAgent: zone.id }); }}>
        {zone.parts.map((p, i) => (
          <Model key={i} url={p.url} size={p.size} position={p.offset ?? [0, 0, 0]} rotation={p.rotation} />
        ))}
      </group>
      <Html position={[0, 5.6, 0]} center distanceFactor={21} zIndexRange={[20, 0]}>
        <div className={`zone-card ${selected ? "selected" : ""} ${inRun ? "" : "muted"}`}
          onClick={(e) => { e.stopPropagation(); set({ selectedAgent: zone.id }); }}   // don't let the canvas treat it as a miss
          style={{ ["--accent" as string]: zone.color }}>
          <div className="zone-card-head">
            <span className="dot" style={{ background: STATUS_COLOR[status], boxShadow: status !== "idle" ? `0 0 8px ${STATUS_COLOR[status]}` : "none" }} />
            <span className="zone-name">{zone.name}</span>
          </div>
          <div className="zone-sub">{inRun ? STATUS_LABEL[status] : "No machines in this scenario"}</div>
          {agent && (
            <div className="zone-stats">
              <span>{agent.writes}<small> writes</small></span>
              <span>{agent.reads}<small> reads</small></span>
              {agent.calls > 0 && <span>{agent.inputTokens.toLocaleString()}<small> tok</small></span>}
            </div>
          )}
        </div>
      </Html>
    </group>
  );
}
