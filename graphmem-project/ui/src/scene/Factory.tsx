import { useEffect, useMemo, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Grid, OrbitControls, Sparkles } from "@react-three/drei";
import { Bloom, EffectComposer, Vignette } from "@react-three/postprocessing";
import { Color, Group } from "three";
import { ZONES } from "../layout";
import { useStore } from "../store";
import { Lanes, Packets } from "./Flows";
import { Hub } from "./Hub";
import { Zone } from "./Zone";

const W = 74, D = 50;

function Ground() {
  const edge = useMemo(() => new Color("#2ee6d0").multiplyScalar(1.6), []);
  const t = 0.22;
  return (
    <group>
      <mesh position={[0, -0.5, 0]} receiveShadow>
        <boxGeometry args={[W, 1, D]} />
        <meshStandardMaterial color="#0a1322" roughness={0.9} metalness={0.2} />
      </mesh>
      <mesh position={[0, -1.4, 0]}>
        <boxGeometry args={[W + 1.5, 0.8, D + 1.5]} />
        <meshStandardMaterial color="#060c18" roughness={1} />
      </mesh>
      {[[0, -D / 2, W, t], [0, D / 2, W, t], [-W / 2, 0, t, D], [W / 2, 0, t, D]].map(([x, z, w, d], i) => (
        <mesh key={i} position={[x, 0.02, z]}>
          <boxGeometry args={[w, 0.08, d]} />
          <meshBasicMaterial color={edge} toneMapped={false} />
        </mesh>
      ))}
      <Grid position={[0, 0.011, 0]} args={[W, D]} cellSize={2} cellThickness={0.6} cellColor="#10233a"
        sectionSize={10} sectionThickness={1} sectionColor="#16385a" fadeDistance={200} infiniteGrid={false} />
    </group>
  );
}

// A few automated vehicles circling the plant perimeter.
function Vehicles() {
  const g = useRef<Group>(null);
  const X = 33, Z = 21.5;
  const perimeter = 2 * (2 * X + 2 * Z);
  useFrame(({ clock }) => {
    g.current?.children.forEach((c, i) => {
      let s = ((clock.elapsedTime * 3.4 + (i * perimeter) / 4) % perimeter + perimeter) % perimeter;
      let x: number, z: number, r: number;
      if (s < 2 * X) { x = -X + s; z = -Z; r = Math.PI / 2; }
      else if ((s -= 2 * X) < 2 * Z) { x = X; z = -Z + s; r = 0; }
      else if ((s -= 2 * Z) < 2 * X) { x = X - s; z = Z; r = -Math.PI / 2; }
      else { s -= 2 * X; x = -X; z = Z - s; r = Math.PI; }
      c.position.set(x, 0.45, z);
      c.rotation.y = r;
    });
  });
  const headlight = useMemo(() => new Color("#bff7ff").multiplyScalar(4), []);
  return (
    <group ref={g}>
      {[0, 1, 2, 3].map((i) => (
        <group key={i}>
          <mesh castShadow><boxGeometry args={[1.0, 0.55, 1.9]} /><meshStandardMaterial color="#27395a" metalness={0.5} roughness={0.4} /></mesh>
          <mesh position={[0, 0.34, -0.1]}><boxGeometry args={[0.8, 0.25, 0.9]} /><meshStandardMaterial color="#14203a" /></mesh>
          {[-0.3, 0.3].map((x) => <mesh key={x} position={[x, 0, 0.97]}><boxGeometry args={[0.2, 0.14, 0.05]} /><meshBasicMaterial color={headlight} toneMapped={false} /></mesh>)}
        </group>
      ))}
    </group>
  );
}

// Keeps the whole plant in view whatever shape the container is.
function FitCamera() {
  const { camera, size } = useThree();
  useEffect(() => {
    const aspect = size.width / Math.max(1, size.height);
    camera.zoom = Math.min(1.5, Math.max(0.5, aspect / 1.2));
    camera.updateProjectionMatrix();
  }, [camera, size]);
  return null;
}

export function Factory({ interactive = true }: { interactive?: boolean }) {
  const set = useStore((s) => s.set);
  return (
    <Canvas shadows dpr={[1, 1.75]} camera={{ position: [76, 58, 76], fov: 24, near: 1, far: 400 }}
      gl={{ antialias: true }} onPointerMissed={() => set({ selectedAgent: undefined })}>
      <color attach="background" args={["#050a13"]} />
      <fog attach="fog" args={["#050a13", 110, 230]} />
      <ambientLight intensity={0.5} color="#8fa9d4" />
      <hemisphereLight args={["#6f9bd8", "#070d19", 0.55]} />
      <directionalLight position={[26, 40, 18]} intensity={1.5} color="#cfe0ff" castShadow
        shadow-mapSize={[2048, 2048]} shadow-camera-left={-46} shadow-camera-right={46}
        shadow-camera-top={34} shadow-camera-bottom={-34} shadow-camera-near={5} shadow-camera-far={140} />

      <FitCamera />
      <Ground />
      <Lanes />
      <Hub />
      {ZONES.map((z) => <Zone key={z.id} zone={z} />)}
      <Vehicles />
      <Packets />
      <Sparkles count={90} scale={[W, 18, D]} position={[0, 9, 0]} size={3.2} speed={0.25} color="#5ff2e0" opacity={0.55} />

      <OrbitControls target={[0, 2, 0]} enableDamping enablePan={false} enabled={interactive}
        minDistance={70} maxDistance={170} minPolarAngle={0.55} maxPolarAngle={1.2}
        minAzimuthAngle={0.1} maxAzimuthAngle={1.5} />
      <EffectComposer multisampling={0}>
        <Bloom mipmapBlur intensity={1.1} luminanceThreshold={0.9} luminanceSmoothing={0.25} />
        <Vignette eskil={false} offset={0.2} darkness={0.7} />
      </EffectComposer>
    </Canvas>
  );
}
