import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import { Color, Group, Mesh, MeshBasicMaterial } from "three";
import { HUB_POS } from "../layout";
import { useStore } from "../store";
import { Model } from "./Model";

const CYAN = new Color("#22d3ee");

// The shared memory graph: a landing pad with a glowing core that flashes on every write.
export function Hub() {
  const counts = useStore((s) => s.counts);
  const condition = useStore((s) => s.condition);
  const core = useRef<Mesh>(null);
  const rings = useRef<Group>(null);

  useFrame(({ clock }) => {
    const t = clock.elapsedTime;
    const { hubPulse } = useStore.getState();
    const since = performance.now() - hubPulse.at;
    const flash = since >= 0 && since < 700 ? 1 - since / 700 : 0;
    if (core.current) {
      const m = core.current.material as MeshBasicMaterial;
      m.color.copy(CYAN).lerp(new Color(hubPulse.color), flash).multiplyScalar(2.2 + flash * 3 + Math.sin(t * 2) * 0.3);
      core.current.scale.set(1 + flash * 0.25, 1, 1 + flash * 0.25);
    }
    if (rings.current) {
      rings.current.rotation.y = t * 0.4;
      rings.current.children.forEach((r, i) => { r.position.y = 1.4 + i * 1.1 + Math.sin(t * 1.5 + i) * 0.15; });
    }
  });

  return (
    <group position={[HUB_POS[0], 0, HUB_POS[1]]}>
      <Model url="/models/kaykit/space/landingpad_large.gltf" size={8.5} />
      <mesh ref={core} position={[0, 2.4, 0]}>
        <cylinderGeometry args={[0.9, 1.15, 3.6, 32]} />
        <meshBasicMaterial toneMapped={false} color="#22d3ee" transparent opacity={0.9} />
      </mesh>
      <group ref={rings}>
        {[0, 1, 2].map((i) => (
          <mesh key={i} rotation-x={Math.PI / 2}>
            <torusGeometry args={[1.9 - i * 0.25, 0.05, 8, 64]} />
            <meshBasicMaterial toneMapped={false} color={new Color("#22d3ee").multiplyScalar(2)} />
          </mesh>
        ))}
      </group>
      <pointLight position={[0, 3, 0]} intensity={30} distance={14} color="#22d3ee" />
      <Html position={[0, 6.6, 0]} center distanceFactor={21} zIndexRange={[30, 0]}>
        <div className="zone-card hub-card">
          <div className="zone-card-head"><span className="dot" style={{ background: "#22d3ee", boxShadow: "0 0 8px #22d3ee" }} />
            <span className="zone-name">Shared Memory</span></div>
          <div className="zone-sub">{condition === "graph" ? "Graph memory" : `Baseline: ${condition}`}</div>
          <div className="zone-stats">
            <span>{counts.active}<small> active</small></span>
            {condition === "graph" && <span>{counts.merge}<small> merged</small></span>}
            {condition === "graph" && <span>{counts.update}<small> replaced</small></span>}
          </div>
        </div>
      </Html>
    </group>
  );
}
