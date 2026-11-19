import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Color, Group, Mesh, MeshBasicMaterial } from "three";
import { ZONES } from "../layout";
import { useStore, type Packet } from "../store";
import { lane, laneWaypoints } from "./curves";

// Glowing ground lanes that run from every zone to the memory core, with a slow ambient pulse on each.
export function Lanes() {
  const lanes = useMemo(() => ZONES.map((z) => {
    const pts = laneWaypoints(z.id);
    const segs = [];
    for (let i = 0; i < pts.length - 1; i++) {
      const a = pts[i], b = pts[i + 1];
      const len = a.distanceTo(b);
      segs.push({ pos: [(a.x + b.x) / 2, 0.1, (a.z + b.z) / 2] as [number, number, number],
        size: (Math.abs(a.x - b.x) > Math.abs(a.z - b.z) ? [len, 0.04, 0.3] : [0.3, 0.04, len]) as [number, number, number] });
    }
    return { id: z.id, color: new Color(z.color).multiplyScalar(1.1), segs };
  }), []);
  const dots = useRef<Group>(null);
  const curves = useMemo(() => ZONES.map((z) => lane(z.id, "hub")), []);
  useFrame(({ clock }) => {
    dots.current?.children.forEach((d, i) => {
      const k = (clock.elapsedTime * 0.07 + i * 0.173) % 1;
      d.position.copy(curves[i].getPoint(k));
      (d as Mesh).scale.setScalar(0.7 + Math.sin(k * Math.PI) * 0.5);
    });
  });
  return (
    <group>
      {lanes.map((l) => l.segs.map((s, i) => (
        <mesh key={l.id + i} position={s.pos}>
          <boxGeometry args={s.size} />
          <meshBasicMaterial toneMapped={false} color={l.color} transparent opacity={0.4} />
        </mesh>
      )))}
      <group ref={dots}>
        {ZONES.map((z) => (
          <mesh key={z.id}>
            <sphereGeometry args={[0.2, 10, 10]} />
            <meshBasicMaterial toneMapped={false} color={new Color(z.color).multiplyScalar(3)} />
          </mesh>
        ))}
      </group>
    </group>
  );
}

function PacketMesh({ p }: { p: Packet }) {
  const ref = useRef<Mesh>(null);
  const c = useMemo(() => lane(p.from, p.to), [p.from, p.to]);
  const color = useMemo(() => new Color(p.color).multiplyScalar(4), [p.color]);
  useFrame(() => {
    const m = ref.current;
    if (!m) return;
    const k = (performance.now() - p.born) / p.duration;
    m.visible = k >= 0 && k <= 1;
    if (!m.visible) return;
    const e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
    c.getPoint(e, m.position);
    m.position.y = 0.55 + Math.sin(k * Math.PI) * 0.5;
    (m.material as MeshBasicMaterial).opacity = Math.min(1, (1 - k) * 4);
  });
  return (
    <mesh ref={ref} visible={false}>
      <sphereGeometry args={[0.42, 16, 16]} />
      <meshBasicMaterial toneMapped={false} color={color} transparent />
    </mesh>
  );
}

// Fact written (zone -> core), facts delivered (core -> agent), push to an inbox (core -> agent).
export function Packets() {
  const packets = useStore((s) => s.packets);
  return <group>{packets.map((p) => <PacketMesh key={p.id} p={p} />)}</group>;
}
