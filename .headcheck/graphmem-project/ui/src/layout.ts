// Where each agent's building sits on the plant floor (x, z in world units).

export type Vec2 = [number, number];

export interface ZoneLayout {
  id: string;            // agent name used by the pipeline
  name: string;
  subtitle: string;
  kind: "production" | "support";
  pos: Vec2;
  color: string;         // accent colour of the zone
  height: number;        // top of the building, used to place the label
}

export const HUB_POS: Vec2 = [0, 0];

export const ZONES: ZoneLayout[] = [
  { id: "press", name: "Press Shop", subtitle: "Stamping presses", kind: "production", pos: [-17, -11], color: "#5eead4", height: 5 },
  { id: "body", name: "Body Shop", subtitle: "Welding robots", kind: "production", pos: [-5, -14], color: "#7dd3fc", height: 5 },
  { id: "paint", name: "Paint Shop", subtitle: "Booths & curing ovens", kind: "production", pos: [8, -14], color: "#a5b4fc", height: 6 },
  { id: "powertrain", name: "Powertrain", subtitle: "Engine & gearbox machining", kind: "production", pos: [20, -9], color: "#f0abfc", height: 11 },
  { id: "assembly", name: "Final Assembly", subtitle: "Assembly line & AGVs", kind: "production", pos: [-15, 11], color: "#86efac", height: 4.5 },
  { id: "quality", name: "Quality", subtitle: "End-of-line test rig", kind: "production", pos: [0, 14], color: "#fde68a", height: 5 },
  { id: "history", name: "Maintenance", subtitle: "Logs, specs & intervals", kind: "support", pos: [-24, 0], color: "#fdba74", height: 6 },
  { id: "diagnosis", name: "Control Room", subtitle: "Root-cause diagnosis", kind: "support", pos: [23, 3], color: "#f472b6", height: 8 },
  { id: "report", name: "Reports", subtitle: "Final structured reports", kind: "support", pos: [15, 13], color: "#c4b5fd", height: 4 },
];

export const ZONE_BY_ID: Record<string, ZoneLayout> = Object.fromEntries(ZONES.map((z) => [z.id, z]));

// Agents from older single-sensor scenarios are drawn in a production zone.
export function zoneFor(agent: string): ZoneLayout {
  return ZONE_BY_ID[agent] ?? (agent === "sensor" ? ZONE_BY_ID.assembly : ZONE_BY_ID.history);
}
