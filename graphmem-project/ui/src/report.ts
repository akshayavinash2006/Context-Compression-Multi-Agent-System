export interface ParsedReport { machine: string; root_cause: string; action: string; evidence: string[] }

// Reports are stored as a JSON string. Returns undefined for anything that is not a report.
export function parseReport(text: string | undefined): ParsedReport | undefined {
  if (!text) return undefined;
  const m = text.match(/\{[\s\S]*\}/);
  if (!m) return undefined;
  try {
    const d = JSON.parse(m[0]);
    if (typeof d !== "object" || d === null || !("root_cause" in d || "machine" in d)) return undefined;
    return {
      machine: String(d.machine ?? ""),
      root_cause: String(d.root_cause ?? "none"),
      action: String(d.action ?? ""),
      evidence: Array.isArray(d.evidence) ? d.evidence.map((e: unknown) => String(e)) : [],
    };
  } catch {
    return undefined;
  }
}

export const label = (s: string) => s.replace(/_/g, " ");

// One-line version for lists: "rob-11 · misalignment → realign motor coupling".
export function reportLine(r: ParsedReport): string {
  return `${r.machine || "unknown machine"} · ${label(r.root_cause)}${r.action ? ` → ${r.action}` : ""}`;
}

// Text to show for any memory: a report is shown as a sentence, everything else as stored.
export function readable(text: string): string {
  const r = parseReport(text);
  return r ? reportLine(r) : text;
}
