"""Shared dataclasses (plan §7, §9.3)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ReadResult:
    memory_ids: list[str] = field(default_factory=list)
    context_text: str = ""
    tokens_used: int = 0
    scores: dict[str, float] = field(default_factory=dict)
    seeds_used: dict[str, float] = field(default_factory=dict)
    fallback: str | None = None          # None | "vector" | "empty"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ReadResult":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class AgentProfile:
    name: str
    role: str
    profile: dict[str, float] = field(default_factory=dict)   # canonical keyword -> weight
    inbox: list[str] = field(default_factory=list)


ROOT_CAUSES = [
    "bearing_wear", "misalignment", "imbalance", "coolant_failure", "overload",
    "lubrication_failure", "sensor_fault", "belt_wear", "none",
]


ACTIONS = {
    "bearing_wear": "replace spindle bearing",
    "misalignment": "realign motor coupling",
    "imbalance": "rebalance and re-mount fixture",
    "coolant_failure": "repair coolant pump",
    "overload": "reduce cutting load",
    "lubrication_failure": "refill and repair lubrication unit",
    "sensor_fault": "recalibrate vibration sensor",
    "belt_wear": "replace drive belt",
    "none": "no action needed",
}


def format_memory_line(mid: str, agent: str, turn: int, text: str) -> str:
    """One context line, as in plan §9.3."""
    text = " ".join(str(text).split())
    return f"[{mid} | {agent} | t={turn}] {text}"
