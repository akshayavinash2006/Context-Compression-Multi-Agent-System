import { Canvas } from "@react-three/fiber";
import { Grid, OrbitControls } from "@react-three/drei";
import { Bloom, EffectComposer } from "@react-three/postprocessing";
import { SCENERY, ZONES } from "../layout";
import { useStore } from "../store";
import { AssemblyLine } from "./AssemblyLine";
import { Packets, Pipes } from "./Flows";
import { Hub } from "./Hub";
import { Model } from "./Model";
import { Zone } from "./Zone";

export function Factory() {
  const set = useStore((s) => s.set);
  return (
    <Canvas shadows dpr={[1, 2]} camera={{ position: [24, 30, 40], fov: 34, near: 1, far: 300 }}
      onPointerMissed={() => set({ selectedAgent: undefined })}>
      <color attach="background" args={["#0a0f17"]} />
      <fog attach="fog" args={["#0a0f17", 70, 140]} />
      <ambientLight intensity={0.45} color="#9db4d6" />
      <hemisphereLight args={["#7aa2d6", "#0b1220", 0.5]} />
      <directionalLight position={[20, 32, 12]} intensity={1.6} color="#dbe7ff" castShadow
        shadow-mapSize={[2048, 2048]} shadow-camera-left={-40} shadow-camera-right={40}
        shadow-camera-top={40} shadow-camera-bottom={-40} />

      {/* plant floor */}
      <mesh position={[0, -0.4, 3]} receiveShadow>
        <boxGeometry args={[60, 0.8, 46]} />
        <meshStandardMaterial color="#0d1626" metalness={0.2} roughness={0.85} />
      </mesh>
      <Grid position={[0, 0.01, 3]} args={[60, 46]} cellSize={1} cellThickness={0.5} cellColor="#13233a"
        sectionSize={5} sectionThickness={1} sectionColor="#1c3554" fadeDistance={120} infiniteGrid={false} />

      <Pipes />
      <Hub />
      {ZONES.map((z) => <Zone key={z.id} zone={z} />)}
      <AssemblyLine />
      {SCENERY.map((s, i) => <Model key={i} url={s.url} size={s.size} position={s.pos} rotation={s.rot} />)}
      <Packets />

      <OrbitControls target={[0, 0, 1]} enableDamping minDistance={20} maxDistance={95}
        maxPolarAngle={1.2} minPolarAngle={0.35} />
      <EffectComposer>
        <Bloom mipmapBlur intensity={0.9} luminanceThreshold={1} luminanceSmoothing={0.2} />
      </EffectComposer>
    </Canvas>
  );
}
