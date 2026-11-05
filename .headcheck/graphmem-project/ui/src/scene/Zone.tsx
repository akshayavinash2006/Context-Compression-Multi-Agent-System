import { useMemo, useRef, useState } from "react";
import { useFrame } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import { Color, Group, Mesh, MeshBasicMaterial } from "three";
import type { ZoneLayout } from "../layout";
import { STATUS_COLOR, STATUS_LABEL } from "../status";
import { useStore, type AgentStatus } from "../store";
import { Building } from "./Buildings";

const PAD_W = 13, PAD_D = 10;

function Frame({ color, refs }: { color: string; refs: React.RefObject<Group | null> }) {
  const mat = useMemo(() => new Color(color), [color]);
  const t = 0.16;
  return (
    <group ref={refs} position={[0, 0.16, 0]}>
      {[[0, -PAD_D / 2, PAD_W, t], [0, PAD_D / 2, PAD_W, t], [-PAD_W / 2, 0, t, PAD_D], [PAD_W / 2, 0, t, PAD_D]].map(([x, z, w, d], i) => (
        <mesh key={i} position={[x, 0, z]}>
          <boxGeometry args={[w, 0.06, d]} />
          <meshBasicMaterial color={mat} toneMapped={false} />
        </mesh>
      ))}
    </group>
  );
}

export function Zone({ zone }: { zone: ZoneLayout }) {
  const agent = useStore((s) => s.agents[zone.id]);
  const selected = useStore((s) => s.selectedAgent === zone.id);
  const inRun = useStore((s) => s.zonesInRun.length === 0 || zone.kind === "support" || s.zonesInRun.includes(zone.id));
  const set = useStore((s) => s.set);
  const [hover, setHover] = useState(false);
  const frame = useRef<Group>(null);
  const beacon = useRef<Mesh>(null);
  const halo = useRef<Mesh>(null);
  const status: AgentStatus = agent?.status ?? "idle";
  const sc = STATUS_COLOR[status];
  const target = useMemo(() => new Color(status === "idle" ? zone.color : sc), [status, zone.color, sc]);

  // The pad outline breathes in the zone colour and pulses in the status colour whenever the agent is busy.
  useFrame(({ clock }) => {
    const t = clock.elapsedTime;
    const active = status !== "idle";
    const k = (active ? 2.2 + Math.sin(t * 7) * 0.9 : 0.55 + (selected ? 0.9 : 0)) * (inRun ? 1 : 0.35);
    frame.current?.children.forEach((c) => ((c as Mesh).material as MeshBasicMaterial).color.copy(target).multiplyScalar(k));
    if (beacon.current) {
      (beacon.current.material as MeshBasicMaterial).color.copy(target).multiplyScalar(active ? 3 : 1.4);
      beacon.current.scale.setScalar(active ? 1.15 + Math.sin(t * 8) * 0.2 : 1);
    }
    if (halo.current) {
      const m = halo.current.material as MeshBasicMaterial;
      const ph = (t * (active ? 1.2 : 0.35)) % 1;
      halo.current.scale.setScalar(1 + ph * 3);
      m.opacity = (1 - ph) * (active ? 0.8 : 0.25);
      m.color.copy(target).multiplyScalar(2);
    }
  });

  const [x, z] = zone.pos;
  const pick = (e: { stopPropagation: () => void }) => { e.stopPropagation(); set({ selectedAgent: selected ? undefined : zone.id }); };
  const open = selected || hover || status !== "idle";
  const over = (e: { stopPropagation: () => void }) => { e.stopPropagation(); setHover(true); document.body.style.cursor = "pointer"; };
  const out = () => { setHover(false); document.body.style.cursor = ""; };

  return (
    <group position={[x, 0, z]}>
      <mesh position={[0, 0.07, 0]} receiveShadow onClick={pick} onPointerOver={over} onPointerOut={out}>
        <boxGeometry args={[PAD_W, 0.14, PAD_D]} />
        <meshStandardMaterial color={selected ? "#14243b" : "#0e182b"} roughness={0.8} metalness={0.2} />
      </mesh>
      <Frame color={zone.color} refs={frame} />
      <group position={[0, 0.14, 0]} onClick={pick} onPointerOver={over} onPointerOut={out} scale={inRun ? 1 : 0.98}>
        <Building id={zone.id} color={zone.color} />
      </group>

      {/* status beacon on the front corner */}
      <group position={[PAD_W / 2 - 0.6, 0, PAD_D / 2 - 0.6]}>
        <mesh position={[0, 0.9, 0]}>
          <cylinderGeometry args={[0.07, 0.07, 1.8, 8]} />
          <meshStandardMaterial color="#35496a" />
        </mesh>
        <mesh ref={beacon} position={[0, 2, 0]}>
          <sphereGeometry args={[0.32, 16, 16]} />
          <meshBasicMaterial toneMapped={false} color={zone.color} />
        </mesh>
        <mesh ref={halo} rotation-x={-Math.PI / 2} position={[0, 0.2, 0]}>
          <ringGeometry args={[0.45, 0.58, 32]} />
          <meshBasicMaterial toneMapped={false} transparent color={zone.color} />
        </mesh>
      </group>

      <Html position={[0, zone.height + 3.4, 0]} center distanceFactor={36} zIndexRange={[20, 0]} style={{ pointerEvents: "none" }}>
        <div className={`sl ${open ? "open" : ""} ${inRun ? "" : "muted"}`} style={{ ["--accent" as string]: zone.color }}>
          <button className="sl-head" style={{ pointerEvents: "auto" }} onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)} onClick={() => set({ selectedAgent: selected ? undefined : zone.id })}>
            <span className="sl-dot" style={{ background: sc, boxShadow: status !== "idle" ? `0 0 10px ${sc}` : "none" }} />
            <span className="sl-name">{zone.name}</span>
          </button>
          <div className="sl-body">
            <div className="sl-sub">{zone.subtitle}</div>
            <div className="sl-status">{inRun ? STATUS_LABEL[status] : "No machines in this scenario"}</div>
            {agent && (
              <div className="sl-stats">
                <span><b>{agent.writes}</b> writes</span>
                <span><b>{agent.reads}</b> reads</span>
                {agent.calls > 0 && <span><b>{agent.inputTokens.toLocaleString()}</b> tok</span>}
              </div>
            )}
            {agent?.lastText && <p className="sl-last">{agent.lastText}</p>}
          </div>
          <i className="sl-line" />
        </div>
      </Html>
    </group>
  );
}
