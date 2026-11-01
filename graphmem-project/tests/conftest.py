import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graphmem.config import load_config  # noqa: E402
from graphmem.memory import GraphMemory  # noqa: E402

LIGHT = {"runtime.use_heavy_models": False, "runtime.llm_cache": None}


@pytest.fixture(scope="session")
def cfg():
    return load_config(overrides=LIGHT)


@pytest.fixture
def mem(cfg):
    return GraphMemory(cfg)


WORKED = {
    "m1": "Hey team, CNC-3 spindle vibration hit 7.2 mm/s at 14:05, above the 6 mm/s limit.",
    "m2": "CNC-3 spindle bearing was last replaced 14 months ago; the service interval is 12 months.",
    "m3": "CNC-5 spindle vibration 3.1 mm/s at 14:10, within the 6 mm/s limit, normal.",
    "dup": "Second report on CNC-3: vibration measured 7.2 mm/s at 14:05, over the 6 mm/s limit.",
    "unrelated": "Material delivery for job 5120 arrived at the stores.",
}
