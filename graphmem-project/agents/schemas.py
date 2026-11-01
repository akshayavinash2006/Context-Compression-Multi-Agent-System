"""Structured output schema for the Report agent (plan §13.3)."""
from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field

RootCause = Literal["bearing_wear", "misalignment", "imbalance", "coolant_failure", "overload",
                    "lubrication_failure", "sensor_fault", "belt_wear", "none"]


class ReportSchema(BaseModel):
    """Final structured maintenance report."""

    machine: str = Field(description="Machine ID in lowercase with a hyphen, e.g. 'cnc-3'")
    root_cause: RootCause = Field(description="One root-cause label from the fixed set")
    action: str = Field(description="Recommended maintenance action")
    evidence: list[str] = Field(default_factory=list, description="IDs of the facts used, e.g. ['M1', 'M2']")


def parse_report_text(text: str) -> ReportSchema | None:
    """Fallback for models without structured output: pull the first JSON object out of text."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    rc = str(data.get("root_cause", "none")).strip().lower().replace(" ", "_")
    data["root_cause"] = rc if rc in RootCause.__args__ else "none"
    data["machine"] = str(data.get("machine", "")).lower().replace(" ", "-")
    data["evidence"] = [str(e).strip("[]") for e in data.get("evidence", []) or []]
    data["action"] = str(data.get("action", ""))
    try:
        return ReportSchema(**{k: data[k] for k in ("machine", "root_cause", "action", "evidence")})
    except Exception:
        return None
