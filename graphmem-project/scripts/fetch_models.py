"""Download the KayKit 3D models used by the factory UI (CC0, by Kay Lousberg, kaylousberg.com).

python scripts/fetch_models.py      ->  ui/public/models/kaykit/{space,city}/*.gltf (+ .bin, textures)
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "ui" / "public" / "models" / "kaykit"
PACKS = {
    "space": ("KayKit-Space-Base-Bits-1.0", "addons/kaykit_space_base_bits/Assets/gltf", [
        "basemodule_A", "basemodule_B", "basemodule_C", "basemodule_D", "basemodule_E", "basemodule_garage",
        "cargodepot_A", "cargodepot_B", "containers_A", "containers_B", "structure_tall", "structure_low",
        "landingpad_large", "drill_structure", "lights", "roofmodule_solarpanels", "solarpanel", "spacetruck",
        "cargo_A_stacked",
    ]),
    "city": ("KayKit-City-Builder-Bits-1.0", "addons/kaykit_city_builder_bits/Assets/gltf", [
        "car_sedan", "car_hatchback", "car_stationwagon", "car_taxi", "watertower", "box_A", "box_B",
        "streetlight",
    ]),
}
RAW = "https://raw.githubusercontent.com/KayKit-Game-Assets/{repo}/main/{path}"
LICENSE = """KayKit models by Kay Lousberg (www.kaylousberg.com), licensed CC0 1.0 (public domain).
Space Base Bits 1.0 and City Builder Bits 1.0: https://github.com/KayKit-Game-Assets
"""


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "graphmem-fetch-models/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main():
    total = 0
    for pack, (repo, base, names) in PACKS.items():
        d = OUT / pack
        d.mkdir(parents=True, exist_ok=True)
        for name in names:
            gltf_path = d / f"{name}.gltf"
            if not gltf_path.exists():
                gltf_path.write_bytes(fetch(RAW.format(repo=repo, path=f"{base}/{name}.gltf")))
            doc = json.loads(gltf_path.read_text(encoding="utf-8"))
            deps = [b.get("uri") for b in doc.get("buffers", [])] + [i.get("uri") for i in doc.get("images", [])]
            for uri in filter(None, deps):
                if uri.startswith("data:"):
                    continue
                target = d / uri
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(fetch(RAW.format(repo=repo, path=f"{base}/{uri}")))
            total += 1
            print(f"  {pack}/{name}")
    (OUT / "LICENSE.txt").write_text(LICENSE, encoding="utf-8")
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"{total} models in {OUT} ({size:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
