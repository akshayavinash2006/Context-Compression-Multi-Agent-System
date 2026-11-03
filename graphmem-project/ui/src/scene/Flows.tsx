import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Color, Mesh, MeshBasicMaterial, TubeGeometry } from "three";
import { ZONES } from "../layout";
import { useStore, type Packet } from "../store";
import { curve } from "./curves";

// Static glowing links from every zone to the memory hub.
export function Pipes() {
  const tubes = useMemo(() => ZONES.map((z) => ({ id: z.id, color: z.color, geo: new TubeGeometry(curve(z.id, "hub"), 48, 0.07, 8, false) })), []);
  return (
    <group>
      {tubes.map((t) => (
        <mesh key={t.id} geometry={t.geo}>
          <meshBasicMaterial toneMapped={false} color={new Color(t.color).multiplyScalar(0.9)} transparent opacity={0.35} />
        </mesh>
      ))}
    </group>
  );
}

function PacketMesh({ p }: { p: Packet }) {
  const ref = useRef<Mesh>(null);
  const c = useMemo(() => curve(p.from, p.to), [p.from, p.to]);
  const color = useMemo(() => new Color(p.color).multiplyScalar(4), [p.color]);
  useFrame(() => {
    const m = ref.current;
    if (!m) return;
    const k = (performance.now() - p.born) / p.duration;
    m.visible = k >= 0 && k <= 1;
    if (!m.visible) return;
    const e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;      // ease in-out
    m.position.copy(c.getPoint(e));
    (m.material as MeshBasicMaterial).opacity = Math.min(1, (1 - k) * 4);
  });
  return (
    <mesh ref={ref} visible={false}>
      <sphereGeometry args={[0.28, 16, 16]} />
      <meshBasicMaterial toneMapped={false} color={color} transparent />
    </mesh>
  );
}

// Moving packets: fact written (zone -> hub), facts delivered (hub -> agent), push to inbox (hub -> agent).
export function Packets() {
  const packets = useStore((s) => s.packets);
  return <group>{packets.map((p) => <PacketMesh key={p.id} p={p} />)}</group>;
}
