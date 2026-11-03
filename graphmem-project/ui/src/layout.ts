// Where each agent's building sits on the plant floor, and which KayKit models make it up.

export type Vec2 = [number, number];

export interface ModelPart {
  url: string;
  size: number;          // target footprint (largest horizontal side), world units
  offset?: [number, number, number];
  rotation?: number;     // around Y, radians
}

export interface ZoneLayout {
  id: string;            // agent name used by the pipeline
  name: string;
  subtitle: string;
  kind: "production" | "support";
  pos: Vec2;             // x, z
  color: string;         // accent colour of the zone
  parts: ModelPart[];
}

const S = (n: string) => `/models/kaykit/space/${n}.gltf`;
const C = (n: string) => `/models/kaykit/city/${n}.gltf`;

export const HUB_POS: Vec2 = [0, 0];

export const ZONES: ZoneLayout[] = [
  { id: "press", name: "Press Shop", subtitle: "Stamping presses", kind: "production", pos: [-15, -11],
    color: "#5eead4",
    parts: [{ url: S("basemodule_garage"), size: 6.5 }, { url: S("containers_A"), size: 2.6, offset: [4.4, 0, 1.6] }] },
  { id: "body", name: "Body Shop", subtitle: "Welding robots", kind: "production", pos: [-5, -13],
    color: "#7dd3fc",
    parts: [{ url: S("basemodule_B"), size: 5.5 }, { url: S("structure_tall"), size: 2.4, offset: [3.6, 0, -1.4] }] },
  { id: "paint", name: "Paint Shop", subtitle: "Booths & curing ovens", kind: "production", pos: [5, -13],
    color: "#a5b4fc",
    parts: [{ url: S("basemodule_C"), size: 5.5 }, { url: C("watertower"), size: 2.2, offset: [-3.6, 0, -1.2] }] },
  { id: "powertrain", name: "Powertrain", subtitle: "Engine & gearbox machining", kind: "production", pos: [15, -11],
    color: "#f0abfc",
    parts: [{ url: S("basemodule_D"), size: 5.5 }, { url: S("drill_structure"), size: 3, offset: [-3.8, 0, 1.6] }] },
  { id: "assembly", name: "Final Assembly", subtitle: "Assembly line & AGVs", kind: "production", pos: [-7, 12],
    color: "#86efac",
    parts: [{ url: S("basemodule_E"), size: 6 }, { url: S("spacetruck"), size: 2.2, offset: [-4.4, 0, 1.2], rotation: 0.6 }] },
  { id: "quality", name: "Quality", subtitle: "End-of-line test rig", kind: "production", pos: [7, 12],
    color: "#fde68a",
    parts: [{ url: S("basemodule_A"), size: 5 }, { url: S("lights"), size: 1.6, offset: [3.2, 0, 1.8] }] },
  { id: "history", name: "Maintenance", subtitle: "Logs, specs & intervals", kind: "support", pos: [-18, 1],
    color: "#fdba74",
    parts: [{ url: S("cargodepot_A"), size: 5.5 }, { url: S("cargo_A_stacked"), size: 2, offset: [0, 0, 3.4] }] },
  { id: "diagnosis", name: "Control Room", subtitle: "Root-cause diagnosis", kind: "support", pos: [18, 1],
    color: "#f472b6",
    parts: [{ url: S("structure_low"), size: 4.6 }, { url: S("roofmodule_solarpanels"), size: 3, offset: [0, 0, 3.2] }] },
  { id: "report", name: "Reports", subtitle: "Final structured reports", kind: "support", pos: [18, 12],
    color: "#c4b5fd",
    parts: [{ url: S("basemodule_A"), size: 3.8 }, { url: C("box_A"), size: 1, offset: [2.6, 0, 1.4] }] },
];

export const ZONE_BY_ID: Record<string, ZoneLayout> = Object.fromEntries(ZONES.map((z) => [z.id, z]));

// Agents from older single-sensor scenarios are drawn in a production zone.
export function zoneFor(agent: string): ZoneLayout {
  return ZONE_BY_ID[agent] ?? (agent === "sensor" ? ZONE_BY_ID.assembly : ZONE_BY_ID.history);
}

export const CARS = ["car_sedan", "car_hatchback", "car_stationwagon", "car_taxi"].map(C);
export const SCENERY: { url: string; pos: [number, number, number]; size: number; rot?: number }[] = [
  { url: S("solarpanel"), pos: [-24, 0, -4], size: 2.4 },
  { url: S("solarpanel"), pos: [-24, 0, -8], size: 2.4 },
  { url: S("containers_B"), pos: [24, 0, -6], size: 3, rot: 1.57 },
  { url: S("cargodepot_B"), pos: [-2, 0, 21], size: 3.6 },
  { url: C("streetlight"), pos: [-10, 0, 4], size: 0.8 },
  { url: C("streetlight"), pos: [10, 0, 4], size: 0.8 },
  { url: C("box_B"), pos: [24, 0, 14], size: 1 },
];
