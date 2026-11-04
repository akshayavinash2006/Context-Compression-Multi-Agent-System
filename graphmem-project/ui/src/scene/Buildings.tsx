import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Color, DoubleSide, ExtrudeGeometry, Group, Mesh, MeshBasicMaterial, Shape } from "three";

// Procedural industrial buildings. Every building is centred on the origin, sits on y = 0 and faces +x / +z
// (the sides the camera sees), where the lit window bands are.

const BODY = "#17233a";
const BODY_LIGHT = "#1e2d48";
const ROOF = "#0f1829";
const METAL = "#2a3a57";

function Box({ p = [0, 0, 0], s, c = BODY, rough = 0.55, metal = 0.35 }: { p?: [number, number, number]; s: [number, number, number]; c?: string; rough?: number; metal?: number }) {
  return (
    <mesh position={[p[0], p[1] + s[1] / 2, p[2]]} castShadow receiveShadow>
      <boxGeometry args={s} />
      <meshStandardMaterial color={c} roughness={rough} metalness={metal} />
    </mesh>
  );
}

function Glow({ p, s, color, k = 2.4 }: { p: [number, number, number]; s: [number, number, number]; color: string; k?: number }) {
  const col = useMemo(() => new Color(color).multiplyScalar(k), [color, k]);
  return (
    <mesh position={p}>
      <boxGeometry args={s} />
      <meshBasicMaterial color={col} toneMapped={false} />
    </mesh>
  );
}

// Rows of lit panes on the two camera-facing walls.
function Windows({ w, d, h, color, rows = 2, base = 1 }: { w: number; d: number; h: number; color: string; rows?: number; base?: number }) {
  const panes: React.JSX.Element[] = [];
  const nz = Math.max(2, Math.floor(w / 1.5));
  const nx = Math.max(2, Math.floor(d / 1.5));
  for (let r = 0; r < rows; r++) {
    const y = base + r * ((h - base - 0.8) / Math.max(1, rows));
    for (let i = 0; i < nz; i++) {
      const x = -w / 2 + (i + 0.5) * (w / nz);
      panes.push(<Glow key={`z${r}${i}`} p={[x, y + 0.5, d / 2 + 0.03]} s={[(w / nz) * 0.62, 0.55, 0.06]} color={color} k={1.6 + ((i + r) % 3) * 0.5} />);
    }
    for (let i = 0; i < nx; i++) {
      const z = -d / 2 + (i + 0.5) * (d / nx);
      panes.push(<Glow key={`x${r}${i}`} p={[w / 2 + 0.03, y + 0.5, z]} s={[0.06, 0.55, (d / nx) * 0.62]} color={color} k={1.6 + ((i + r) % 3) * 0.5} />);
    }
  }
  return <>{panes}</>;
}

export function Smoke({ p, count = 7, rate = 1, height = 7, color = "#8fa6c4" }: { p: [number, number, number]; count?: number; rate?: number; height?: number; color?: string }) {
  const g = useRef<Group>(null);
  useFrame(({ clock }) => {
    const t = clock.elapsedTime * 0.28 * rate;
    g.current?.children.forEach((c, i) => {
      const k = (t + i / count) % 1;
      c.position.set(Math.sin(i * 2.1 + k * 3) * 0.5 + k * 1.6, k * height, Math.cos(i * 1.7) * 0.4 + k * 0.8);
      c.scale.setScalar(0.5 + k * 2.1);
      ((c as Mesh).material as MeshBasicMaterial).opacity = (1 - k) * 0.22 * Math.min(1, k * 6);
    });
  });
  return (
    <group ref={g} position={p}>
      {Array.from({ length: count }, (_, i) => (
        <mesh key={i}>
          <sphereGeometry args={[0.7, 10, 10]} />
          <meshBasicMaterial color={color} transparent depthWrite={false} opacity={0} />
        </mesh>
      ))}
    </group>
  );
}

function Stack({ p, h = 8, r = 0.7, accent }: { p: [number, number, number]; h?: number; r?: number; accent: string }) {
  const band = useMemo(() => new Color(accent).multiplyScalar(2.4), [accent]);
  return (
    <group position={p}>
      <mesh position={[0, h / 2, 0]} castShadow>
        <cylinderGeometry args={[r * 0.8, r, h, 20]} />
        <meshStandardMaterial color={METAL} roughness={0.5} metalness={0.5} />
      </mesh>
      <mesh position={[0, h - 0.5, 0]}>
        <cylinderGeometry args={[r * 0.84, r * 0.86, 0.22, 20]} />
        <meshBasicMaterial color={band} toneMapped={false} />
      </mesh>
      <mesh position={[0, h * 0.5, 0]}>
        <cylinderGeometry args={[r * 0.9, r * 0.94, 0.14, 20]} />
        <meshBasicMaterial color={band} toneMapped={false} />
      </mesh>
      <Smoke p={[0, h, 0]} height={6} />
    </group>
  );
}

function Tank({ p, r = 1.6, h = 5, accent, glow = false }: { p: [number, number, number]; r?: number; h?: number; accent: string; glow?: boolean }) {
  const col = useMemo(() => new Color(accent).multiplyScalar(glow ? 2 : 1.2), [accent, glow]);
  return (
    <group position={p}>
      <mesh position={[0, h / 2, 0]} castShadow>
        <cylinderGeometry args={[r, r, h, 28]} />
        <meshStandardMaterial color={glow ? "#0f4d57" : METAL} roughness={0.35} metalness={0.6} emissive={glow ? accent : "#000"} emissiveIntensity={glow ? 0.5 : 0} />
      </mesh>
      <mesh position={[0, h + 0.08, 0]}>
        <cylinderGeometry args={[r * 1.02, r * 1.02, 0.16, 28]} />
        <meshBasicMaterial color={col} toneMapped={false} />
      </mesh>
      <mesh position={[0, h * 0.5, 0]}>
        <cylinderGeometry args={[r * 1.02, r * 1.02, 0.1, 28]} />
        <meshBasicMaterial color={col} toneMapped={false} />
      </mesh>
    </group>
  );
}

// Saw-tooth roof used by the body shop.
function SawRoof({ w, d, y, n, color }: { w: number; d: number; y: number; n: number; color: string }) {
  const geo = useMemo(() => {
    const s = new Shape();
    const seg = d / n;
    s.moveTo(0, 0); s.lineTo(seg, 0); s.lineTo(seg, 1.4); s.lineTo(0, 0);
    return new ExtrudeGeometry(s, { depth: w, bevelEnabled: false });
  }, [w, d, n]);
  const glow = useMemo(() => new Color(color).multiplyScalar(2), [color]);
  return (
    <group position={[-w / 2, y, d / 2]} rotation={[0, Math.PI / 2, 0]}>
      {Array.from({ length: n }, (_, i) => (
        <group key={i} position={[(i * d) / n, 0, 0]}>
          <mesh geometry={geo} position={[0, 0, 0]} rotation={[0, 0, 0]} castShadow>
            <meshStandardMaterial color={ROOF} roughness={0.5} metalness={0.4} side={DoubleSide} />
          </mesh>
          <mesh position={[(d / n) * 0.98, 0.7, w / 2]}>
            <boxGeometry args={[0.05, 1.2, w * 0.9]} />
            <meshBasicMaterial color={glow} toneMapped={false} />
          </mesh>
        </group>
      ))}
    </group>
  );
}

export function Building({ id, color }: { id: string; color: string }) {
  switch (id) {
    case "press":
      return (
        <group>
          <Box s={[9, 4, 6]} c={BODY} />
          <Box p={[0, 4, 0]} s={[9.4, 0.3, 6.4]} c={ROOF} />
          {[-2.8, 0, 2.8].map((x) => <Box key={x} p={[x, 4.3, -0.5]} s={[1.6, 1.1, 2.2]} c={METAL} />)}
          <Windows w={9} d={6} h={4} color={color} />
          <Glow p={[0, 4.35, 3.25]} s={[9.4, 0.12, 0.08]} color={color} k={2.8} />
        </group>
      );
    case "body":
      return (
        <group>
          <Box s={[9, 3.6, 7]} c={BODY} />
          <SawRoof w={9} d={7} y={3.6} n={3} color={color} />
          <Windows w={9} d={7} h={3.6} color={color} rows={1} base={1.2} />
          <Box p={[5.4, 0, 2.4]} s={[1.6, 1.8, 1.6]} c={METAL} />
        </group>
      );
    case "paint":
      return (
        <group>
          <Box s={[8, 3.4, 6]} c={BODY} />
          <Box p={[0, 3.4, 0]} s={[8.4, 0.3, 6.4]} c={ROOF} />
          <Tank p={[-2.2, 3.7, 0]} r={1.2} h={2.2} accent={color} />
          <Tank p={[1.2, 3.7, 0]} r={1.2} h={2.6} accent={color} />
          <Tank p={[5.2, 0, -1.8]} r={1.3} h={5} accent={color} glow />
          <Windows w={8} d={6} h={3.4} color={color} rows={1} base={1.1} />
        </group>
      );
    case "powertrain":
      return (
        <group>
          <Box s={[9, 5, 7]} c={BODY_LIGHT} />
          <Box p={[0, 5, 0]} s={[9.4, 0.3, 7.4]} c={ROOF} />
          <Box p={[-2.6, 5.3, 1]} s={[2.6, 1.4, 2.6]} c={METAL} />
          <Stack p={[-3, 5.3, -2]} h={6.5} r={0.9} accent={color} />
          <Stack p={[0.2, 5.3, -2.4]} h={5} r={0.75} accent={color} />
          <Stack p={[3, 5.3, -2]} h={7.5} r={0.9} accent={color} />
          <Windows w={9} d={7} h={5} color="#fb923c" rows={2} base={1} />
        </group>
      );
    case "assembly":
      return (
        <group>
          <Box s={[11, 3.4, 5.5]} c={BODY} />
          <Box p={[0, 3.4, 0]} s={[11.4, 0.28, 5.9]} c={ROOF} />
          <Glow p={[0, 3.8, 0]} s={[9.6, 0.1, 0.5]} color={color} k={2.6} />
          <Glow p={[0, 3.8, -1.4]} s={[9.6, 0.1, 0.12]} color={color} k={2} />
          <Glow p={[0, 3.8, 1.4]} s={[9.6, 0.1, 0.12]} color={color} k={2} />
          <Windows w={11} d={5.5} h={3.4} color={color} rows={1} base={1.2} />
          <Box p={[-6.4, 0, 2.6]} s={[1.6, 1.2, 1.6]} c={METAL} />
        </group>
      );
    case "quality":
      return (
        <group>
          <Box s={[7.5, 3.2, 6]} c={BODY} />
          <mesh position={[0, 3.2, 0]} castShadow>
            <sphereGeometry args={[2.8, 28, 14, 0, Math.PI * 2, 0, Math.PI / 2]} />
            <meshStandardMaterial color="#0e5360" roughness={0.12} metalness={0.7} emissive={color} emissiveIntensity={0.35} />
          </mesh>
          <Glow p={[0, 3.25, 0]} s={[6.6, 0.1, 5.2]} color={color} k={1.8} />
          <Windows w={7.5} d={6} h={3.2} color={color} rows={1} base={1} />
        </group>
      );
    case "history":
      return (
        <group>
          <Box s={[8, 3, 5]} c={BODY} />
          <Box p={[0, 3, 0]} s={[8.3, 0.25, 5.3]} c={ROOF} />
          <Tank p={[-2.6, 3.2, 0]} r={1} h={2.2} accent={color} />
          <Tank p={[0, 3.2, 0]} r={1} h={2.8} accent={color} />
          <Tank p={[2.6, 3.2, 0]} r={1} h={2.2} accent={color} />
          <Box p={[5.4, 0, 2.2]} s={[1.5, 1.3, 1.5]} c={METAL} />
          <Box p={[5.4, 1.3, 2.2]} s={[1.2, 1.1, 1.2]} c={BODY_LIGHT} />
          <Windows w={8} d={5} h={3} color={color} rows={1} base={1} />
        </group>
      );
    case "diagnosis":
      return (
        <group>
          <Box s={[6, 3, 6]} c={BODY} />
          <Box p={[0, 3, 0]} s={[4, 4.2, 4]} c={BODY_LIGHT} />
          <Box p={[0, 7.2, 0]} s={[4.6, 0.4, 4.6]} c={ROOF} />
          <Glow p={[0, 5.2, 2.03]} s={[3.2, 1.6, 0.06]} color={color} k={2.2} />
          <Glow p={[2.03, 5.2, 0]} s={[0.06, 1.6, 3.2]} color={color} k={2.2} />
          <mesh position={[0, 8.1, 0]} rotation={[-0.7, 0.8, 0]}>
            <sphereGeometry args={[1.2, 18, 8, 0, Math.PI * 2, 0, 1.1]} />
            <meshStandardMaterial color="#8da2bf" roughness={0.3} metalness={0.7} side={DoubleSide} />
          </mesh>
          <Windows w={6} d={6} h={3} color={color} rows={1} base={1} />
        </group>
      );
    default:
      return (
        <group>
          <Box s={[6, 3, 5]} c={BODY} />
          <Box p={[0, 3, 0]} s={[6.3, 0.25, 5.3]} c={ROOF} />
          <Box p={[3.8, 0, 1.6]} s={[1.2, 1, 1.2]} c={METAL} />
          <Windows w={6} d={5} h={3} color={color} rows={1} base={1} />
        </group>
      );
  }
}
