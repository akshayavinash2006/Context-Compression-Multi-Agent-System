import { useEffect, useState } from "react";

export type Route = "overview" | "run" | "analytics" | "activity" | "diagnosis" | "graph" | "agents";

export const ROUTES: { id: Route; label: string; hint: string }[] = [
  { id: "overview", label: "Plant Overview", hint: "Live factory floor" },
  { id: "run", label: "Run", hint: "Configure, start and follow a run" },
  { id: "analytics", label: "Memory Analytics", hint: "Token savings and memory health" },
  { id: "activity", label: "Activity", hint: "Every read, write and model call" },
  { id: "diagnosis", label: "Diagnosis", hint: "Questions, answers and traces" },
  { id: "graph", label: "Memory Graph", hint: "The shared graph, as it grows" },
  { id: "agents", label: "Agents", hint: "Who does what, on which model" },
];

const read = (): Route => {
  const h = location.hash.replace(/^#\/?/, "").split("?")[0] as Route;
  return ROUTES.some((r) => r.id === h) ? h : "overview";
};

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(read);
  useEffect(() => {
    const on = () => setRoute(read());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}

export const go = (r: Route) => { location.hash = `#/${r}`; };
