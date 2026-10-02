"""Run the v1 structural-fact layer on practice cards through build.nvidia (stand-in for the House route).

usage: python backtest/run_nemotron.py N        # N cards, stratified by family, fixed seed; cached per card
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from t2agent.house import House  # noqa: E402
from t2agent.task import load_task  # noqa: E402
from t2agent.textlayer import extract  # noqa: E402

UNITS = ROOT.parents[1] / "track2-forecasting-public" / "units"
DATA = ROOT / "backtest" / "data"
OUT = DATA / "nemotron_v1"


def local_env() -> None:
    key = re.search(r"^NVIDIA_API_KEY=(.*)$", (ROOT.parents[1] / ".env.nvidia").read_text(), re.M).group(1).strip()
    os.environ.setdefault("MODEL_ENDPOINT", "https://integrate.api.nvidia.com")
    os.environ.setdefault("MODEL_NAME", "nvidia/nemotron-3-super-120b-a12b")
    os.environ["MODEL_TOKEN"] = key


def pick(n: int) -> list[str]:
    real = json.loads((DATA / "realized.json").read_text())
    ids = sorted(u for u, r in real.items() if r["complete"])
    rng = np.random.default_rng(11)
    fams = {f: [u for u in ids if real[u]["category"] == f] for f in ("T2-F1", "T2-F2", "T2-F3", "T2-F4")}
    for f in fams:
        rng.shuffle(fams[f])
    out, k = [], 0
    while len(out) < n:
        for f in fams:
            if k < len(fams[f]) and len(out) < n:
                out.append(fams[f][k])
        k += 1
    return out


if __name__ == "__main__":
    local_env()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    OUT.mkdir(parents=True, exist_ok=True)
    real = json.loads((DATA / "realized.json").read_text())
    for uid in pick(n):
        f = OUT / f"{uid}.json"
        if f.exists():
            continue
        t = load_task(UNITS / uid, UNITS / uid / "text", real[uid]["asof"])
        house = House()
        t0 = time.monotonic()
        try:
            res = extract(t, house)
            err = ""
        except Exception as e:
            res, err = None, repr(e)[:200]
        rec = {"uid": uid, "seconds": round(time.monotonic() - t0, 1), "requests": house.used, "log": house.log,
               "ok": res is not None, "error": err, "result": res}
        f.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        toks = sum((e.get("usage") or {}).get("completion_tokens", 0) for e in house.log)
        print(f"{uid}: ok={rec['ok']} {rec['seconds']}s requests={house.used} out_tokens={toks} {err}", flush=True)
