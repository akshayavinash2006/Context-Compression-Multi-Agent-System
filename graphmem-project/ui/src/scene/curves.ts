import { QuadraticBezierCurve3, Vector3 } from "three";
import { HUB_POS, zoneFor } from "../layout";

const cache = new Map<string, QuadraticBezierCurve3>();

function point(id: string, y: number): Vector3 {
  if (id === "hub") return new Vector3(HUB_POS[0], y, HUB_POS[1]);
  const z = zoneFor(id);
  return new Vector3(z.pos[0], y, z.pos[1]);
}

// Arched path between a zone and the memory hub, used by pipes and data packets.
export function curve(from: string, to: string): QuadraticBezierCurve3 {
  const key = `${from}>${to}`;
  let c = cache.get(key);
  if (!c) {
    const a = point(from, 0.35);
    const b = point(to, 0.35);
    const mid = a.clone().lerp(b, 0.5);
    mid.y = 2.2 + a.distanceTo(b) * 0.06;
    c = new QuadraticBezierCurve3(a, mid, b);
    cache.set(key, c);
  }
  return c;
}
