import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import { Color, Group, Mesh, MeshBasicMaterial, MeshStandardMaterial } from "three";
import { HUB_POS } from "../layout";
import { useStore } from "../store";

const TEAL = "#2ee6d0";

// The shared memory graph, drawn as a tall glowing storage tank that flashes on every write.
export function Hub() {
  const counts = useStore((s) => s.counts);
  const condition = useStore((s) => s.condition);
  const tank = useRef<Mesh>(null);
  const rings = useRef<Group>(null);
  const halo = useRef<Mesh>(null);
  const base = useMemo(() => new Color(TEAL), []);
  const tmp = useMemo(() => new Color(), []);

  useFrame(({ clock }) => {
    const t = clock.elapsedTime;
    const { hubPulse } = useStore.getState();
    const since = performance.now() - hubPulse.at;
    const flash = since >= 0 && since < 800 ? 1 - since / 800 : 0;
    if (tank.current) {
      const m = tank.current.material as MeshStandardMaterial;
      tmp.copy(base).lerp(new Color(hubPulse.color), flash);
      m.emissive.copy(tmp);
      m.emissiveIntensity = 0.55 + flash * 1.6 + Math.sin(t * 1.6) * 0.08;
    }
    if (rings.current) {
      rings.current.children.forEach((r, i) => {
        r.position.y = 1.2 + ((t * 0.9 + i * 0.34) % 1) * 9.6;
        const k = (t * 0.9 + i * 0.34) % 1;
        const mat = (r as Mesh).material as MeshBasicMaterial;
        mat.opacity = Math.sin(k * Math.PI) * (0.55 + flash * 0.4);
        r.scale.setScalar(1 + flash * 0.1);
      });
    }
    if (halo.current) {
      const ph = (t * 0.5) % 1;
      halo.current.scale.setScalar(1 + ph * 2.2 + flash * 0.4);
      (halo.current.material as MeshBasicMaterial).opacity = (1 - ph) * 0.45;
    }
  });

  const glow = useMemo(() => new Color(TEAL).multiplyScalar(2.4), []);
  return (
    <group position={[HUB_POS[0], 0, HUB_POS[1]]}>
      <mesh position={[0, 0.12, 0]} receiveShadow>
        <cylinderGeometry args={[5.4, 5.6, 0.24, 48]} />
        <meshStandardMaterial color="#0c1628" roughness={0.7} metalness={0.3} />
      </mesh>
      <mesh rotation-x={-Math.PI / 2} position={[0, 0.27, 0]}>
        <ringGeometry args={[5.1, 5.35, 64]} />
        <meshBasicMaterial color={glow} toneMapped={false} />
      </mesh>
      <mesh ref={halo} rotation-x={-Math.PI / 2} position={[0, 0.3, 0]}>
        <ringGeometry args={[2.6, 2.75, 64]} />
        <meshBasicMaterial color={glow} toneMapped={false} transparent />
      </mesh>

      {/* main glowing tank */}
      <mesh ref={tank} position={[0, 5.9, 0]} castShadow>
        <cylinderGeometry args={[2, 2, 11.2, 40]} />
        <meshStandardMaterial color="#0f5560" emissive={TEAL} emissiveIntensity={0.6} roughness={0.2} metalness={0.5} transparent opacity={0.92} />
      </mesh>
      {[0.5, 3.3, 6.3, 9.2, 11.4].map((y) => (
        <mesh key={y} position={[0, y + 0.1, 0]}>
          <cylinderGeometry args={[2.06, 2.06, 0.12, 40]} />
          <meshBasicMaterial color={glow} toneMapped={false} />
        </mesh>
      ))}
      <mesh position={[0, 11.6, 0]}>
        <cylinderGeometry args={[1.3, 2, 0.5, 40]} />
        <meshStandardMaterial color="#17324a" roughness={0.3} metalness={0.7} />
      </mesh>
      <group ref={rings}>
        {[0, 1, 2].map((i) => (
          <mesh key={i} rotation-x={Math.PI / 2}>
            <torusGeometry args={[2.35, 0.05, 8, 64]} />
            <meshBasicMaterial color={glow} toneMapped={false} transparent />
          </mesh>
        ))}
      </group>

      {/* grey sibling tank */}
      <mesh position={[3.9, 4.2, -2.2]} castShadow>
        <cylinderGeometry args={[1.6, 1.6, 8.4, 32]} />
        <meshStandardMaterial color="#8da2bf" roughness={0.28} metalness={0.75} />
      </mesh>
      <mesh position={[3.9, 8.5, -2.2]}>
        <sphereGeometry args={[1.6, 24, 10, 0, Math.PI * 2, 0, Math.PI / 2]} />
        <meshStandardMaterial color="#a9bad2" roughness={0.25} metalness={0.8} />
      </mesh>
      <mesh position={[0.9, 0.75, 0.9]} castShadow>
        <boxGeometry args={[1.2, 1.5, 1.2]} />
        <meshStandardMaterial color="#1e2d48" />
      </mesh>

      <pointLight position={[0, 6, 0]} intensity={140} distance={26} color={TEAL} />

      <Html position={[0, 14.4, 0]} center distanceFactor={36} zIndexRange={[30, 0]} style={{ pointerEvents: "none" }}>
        <div className="sl open hub" style={{ ["--accent" as string]: TEAL }}>
          <div className="sl-head static"><span className="sl-dot" style={{ background: TEAL, boxShadow: `0 0 10px ${TEAL}` }} /><span className="sl-name">Shared Memory</span></div>
          <div className="sl-body">
            <div className="sl-status">{condition === "graph" ? "Graph memory" : `Baseline · ${condition}`}</div>
            <div className="sl-stats">
              <span><b>{counts.active}</b> active</span>
              {condition === "graph" && <span><b>{counts.merge}</b> merged</span>}
              {condition === "graph" && <span><b>{counts.update}</b> replaced</span>}
            </div>
          </div>
          <i className="sl-line" />
        </div>
      </Html>
    </group>
  );
}
