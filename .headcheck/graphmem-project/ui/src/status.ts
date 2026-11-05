import type { AgentStatus } from "./store";

export const STATUS_COLOR: Record<AgentStatus, string> = {
  idle: "#4b6a86", reading: "#fbbf24", thinking: "#a78bfa", writing: "#2ee6d0",
};
export const STATUS_LABEL: Record<AgentStatus, string> = {
  idle: "Standing by", reading: "Reading memory", thinking: "Calling LLM", writing: "Writing to memory",
};
