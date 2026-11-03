"""Role prompts as LangChain ChatPromptTemplates (plan §13.2-13.3)."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from graphmem.models import ROOT_CAUSES

ROLES = {
    "sensor": {
        "role": "Sensor",
        "description": "You read machine sensor data and operator notes and report anomalies "
                       "with exact values, units, limits and times.",
        "profile": ["vibration", "temperature", "current", "pressure", "limit", "threshold",
                    "speed fluctuation"],
    },
    "history": {
        "role": "History",
        "description": "You keep track of maintenance logs and specifications: replacements, "
                       "service dates and service intervals.",
        "profile": ["maintenance", "replaced", "interval", "service", "spindle bearing", "bearing",
                    "coolant pump", "belt", "lubrication", "calibration", "coupling"],
    },
    "diagnosis": {
        "role": "Diagnosis",
        "description": "You find the most likely root cause of a machine problem by combining "
                       "sensor anomalies with maintenance history, and recommend an action.",
        "profile": ["vibration", "overheating", "temperature", "noise", "current", "wear",
                    "misalignment", "imbalance", "spindle bearing", "bearing", "coolant pump",
                    "belt", "lubrication", "calibration", "overdue", "interval", "replaced"],
    },
    "report": {
        "role": "Report",
        "description": "You write the final structured maintenance report for the plant manager.",
        "profile": ["root cause", "action", "overdue"],
    },
}

def role_for(agent: str, zones: dict | None = None) -> dict:
    """Role of any agent: the four core roles, or a zone monitoring agent (car plant)."""
    if agent in ROLES:
        return ROLES[agent]
    z = (zones or {}).get(agent, {})
    name = z.get("name", agent.title())
    return {
        "role": f"{name} monitoring",
        "description": f"You monitor the {name} ({z.get('description', 'a production zone')}). You read its "
                       "machine sensor data and operator notes and report anomalies with exact values, "
                       "units, limits and times.",
        "profile": ROLES["sensor"]["profile"] + list(z.get("machines", [])),
    }


def task_kind(agent: str) -> str:
    """Which task template an agent uses: zone agents behave like the Sensor agent."""
    return agent if agent in ("history", "diagnosis", "report") else "sensor"


SYSTEM = (
    "You are the {role} agent in a factory maintenance team. {role_description}\n"
    "Use only the facts in MEMORY. Each fact starts with an ID like [M12].\n"
    "Cite the IDs of every fact you use, in square brackets.\n"
    "If a fact looks incomplete and you need its full original text, call the expand_memory tool."
)

AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "MEMORY:\n{context}\n\nTASK:\n{task}"),
    MessagesPlaceholder("scratch", optional=True),
])

DIAGNOSIS_TASK = (
    "{question}\n"
    "Answer in two sentences: the most likely root cause for the machine, citing fact IDs, "
    "then the recommended action."
)

REPORT_TASK = (
    "Question: {question}\n"
    "Write the final report as JSON with fields: machine (e.g. 'cnc-3'), root_cause (one of: "
    + ", ".join(ROOT_CAUSES) + "), action, evidence (list of fact IDs you relied on)."
)

SENSOR_TASK = "New input from the shop floor:\n{text}\nReport it concisely, keeping all values, units and times."
HISTORY_TASK = "New maintenance/spec record:\n{text}\nRecord the key facts concisely, keeping all numbers."

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You maintain a running summary of a factory maintenance team's conversation. "
               "Keep every machine ID, number, unit, limit and maintenance fact. Be concise."),
    ("human", "SUMMARISE\nCURRENT SUMMARY:\n{summary}\n\nNEW MESSAGES:\n{messages}\n\n"
              "Write the updated summary."),
])
