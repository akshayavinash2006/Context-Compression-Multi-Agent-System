import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Color, Group } from "three";
import { CARS } from "../layout";
import { useStore } from "../store";
import { Model } from "./Model";

const START = -16, END = 2, Z = 18.2;
const SLOTS = 5;

// Conveyor in front of Final Assembly; cars move while a run is in progress.
export function AssemblyLine() {
  const cars = useRef<Group>(null);
  const offset = useRef(0);
  useFrame((_, dt) => {
    const running = useStore.getState().wsStatus === "running";
    if (running) offset.current = (offset.current + dt * 0.9) % ((END - START) / SLOTS);
    cars.current?.children.forEach((c, i) => {
      const span = END - START;
      const x = START + ((i * span) / SLOTS + offset.current) % span;
      c.position.x = x;
    });
  });
  return (
    <group>
      <mesh position={[(START + END) / 2, 0.12, Z]} receiveShadow>
        <boxGeometry args={[END - START + 2, 0.24, 1.8]} />
        <meshStandardMaterial color="#1e293b" metalness={0.4} roughness={0.6} />
      </mesh>
      {[-0.95, 0.95].map((dz) => (
        <mesh key={dz} position={[(START + END) / 2, 0.26, Z + dz]}>
          <boxGeometry args={[END - START + 2, 0.06, 0.08]} />
          <meshBasicMaterial toneMapped={false} color={new Color("#86efac").multiplyScalar(1.6)} />
        </mesh>
      ))}
      <group ref={cars}>
        {Array.from({ length: SLOTS }, (_, i) => (
          <group key={i} position={[START, 0.24, Z]}>
            <Model url={CARS[i % CARS.length]} size={1.8} rotation={Math.PI / 2} />
          </group>
        ))}
      </group>
    </group>
  );
}
