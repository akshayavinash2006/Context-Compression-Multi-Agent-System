import { CurvePath, LineCurve3, Vector3 } from "three";
import { HUB_POS, zoneFor } from "../layout";

const cache = new Map<string, CurvePath<Vector3>>();
const LANE_Y = 0.12;

function point(id: string): Vector3 {
  if (id === "hub") return new Vector3(HUB_POS[0], LANE_Y, HUB_POS[1]);
  const z = zoneFor(id);
  return new Vector3(z.pos[0], LANE_Y, z.pos[1]);
}

// Right-angled route along the ground between a zone and the memory core: first to the hub row, then across.
export function laneWaypoints(zoneId: string): Vector3[] {
  const a = point(zoneId);
  const h = point("hub");
  const corner = new Vector3(a.x, LANE_Y, h.z);
  const pts = [a];
  if (Math.abs(corner.z - a.z) > 0.01) pts.push(corner);
  pts.push(h);
  return pts;
}

export function lane(from: string, to: string): CurvePath<Vector3> {
  const key = `${from}>${to}`;
  let c = cache.get(key);
  if (!c) {
    const zoneId = from === "hub" ? to : from;
    let pts = laneWaypoints(zoneId);
    if (from === "hub") pts = [...pts].reverse();
    c = new CurvePath<Vector3>();
    for (let i = 0; i < pts.length - 1; i++) c.add(new LineCurve3(pts[i], pts[i + 1]));
    cache.set(key, c);
  }
  return c;
}
