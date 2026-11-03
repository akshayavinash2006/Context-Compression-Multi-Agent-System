import { Component, ReactNode, Suspense, useMemo } from "react";
import { useGLTF } from "@react-three/drei";
import { Box3, Group, Mesh, Vector3 } from "three";

interface Props { url: string; size: number; position?: [number, number, number]; rotation?: number }

// Loads a glTF, scales it so its largest horizontal side equals `size`, and sits it on the ground.
function Loaded({ url, size, position = [0, 0, 0], rotation = 0 }: Props) {
  const { scene } = useGLTF(url);
  const obj = useMemo(() => {
    const c = scene.clone(true);
    const box = new Box3().setFromObject(c);
    const dim = box.getSize(new Vector3());
    c.scale.setScalar(size / Math.max(dim.x, dim.z, 1e-3));
    const b2 = new Box3().setFromObject(c);
    const center = b2.getCenter(new Vector3());
    c.position.set(-center.x, -b2.min.y, -center.z);
    c.traverse((o) => {
      if ((o as Mesh).isMesh) { o.castShadow = true; o.receiveShadow = true; }
    });
    const g = new Group();
    g.add(c);
    return g;
  }, [scene, size]);
  return <primitive object={obj} position={position} rotation-y={rotation} />;
}

// Grey block shown while a model loads, or if its file is missing.
function Placeholder({ size, position = [0, 0, 0] }: Props) {
  return (
    <mesh position={[position[0], size * 0.3, position[2]]} castShadow>
      <boxGeometry args={[size * 0.8, size * 0.6, size * 0.8]} />
      <meshStandardMaterial color="#334155" />
    </mesh>
  );
}

class Boundary extends Component<{ fallback: ReactNode; children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() { return this.state.failed ? this.props.fallback : this.props.children; }
}

export function Model(p: Props) {
  return (
    <Boundary fallback={<Placeholder {...p} />}>
      <Suspense fallback={<Placeholder {...p} />}>
        <Loaded {...p} />
      </Suspense>
    </Boundary>
  );
}
