"""Offline: Nemotron tone of every FOMC statement 2000-2024 via first-token probabilities (Step 1b protocol).
n1 = E[digit] for "how hawkish" (1..5); n2 = P(A: more hawkish) - P(C: more dovish) vs the previous statement."""

from __future__ import annotations

import json
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.run_nemotron import local_env  # noqa: E402
from t2agent.house import House  # noqa: E402

EXT = ROOT / "backtest" / "data" / "external" / "fed"
OUT = ROOT / "backtest" / "data" / "nemotron_tone.json"
SYS = "You are an expert on Federal Reserve communication. Judge only the text you are given. Answer with one character."
Q1 = """FOMC statement:
<<<
{s}
>>>
On a scale from 1 to 5, how hawkish is this statement's monetary policy stance and guidance?
1 = very dovish (easing, strong accommodation), 3 = neutral, 5 = very hawkish (tightening, inflation-fighting).
Answer with a single digit."""
Q2 = """Previous FOMC statement:
<<<
{p}
>>>
Current FOMC statement:
<<<
{s}
>>>
Compared with the previous statement, is the current statement's policy stance and guidance
A = more hawkish, B = about the same, C = more dovish? Answer with a single letter."""


def probs(q: str) -> dict:
    """build.nvidia rate-limits (~40 rpm): pace calls and back off on 429. (The House route has no rpm limit.)"""
    import time
    for attempt in range(6):
        time.sleep(1.6)
        try:
            return House(max_requests=2).first_token_probs([{"role": "system", "content": SYS}, {"role": "user", "content": q}])
        except Exception as e:
            if "429" not in repr(e) or attempt == 5:
                raise
            time.sleep(10 * (attempt + 1))


def score(item):
    date, cur, prev = item
    out = {"date": date}
    try:
        p = probs(Q1.format(s=cur))
        tot = sum(p.get(str(i), 0) for i in range(1, 6))
        out["n1"] = sum(i * p.get(str(i), 0) for i in range(1, 6)) / tot if tot > 0 else None
        out["n1_mass"] = tot
    except Exception as e:
        out["n1"], out["err1"] = None, repr(e)[:120]
    if prev:
        try:
            p = probs(Q2.format(p=prev, s=cur))
            tot = p.get("A", 0) + p.get("B", 0) + p.get("C", 0)
            out["n2"] = (p.get("A", 0) - p.get("C", 0)) / tot if tot > 0 else None
            out["n2_mass"] = tot
        except Exception as e:
            out["n2"], out["err2"] = None, repr(e)[:120]
    return out


if __name__ == "__main__":
    local_env()
    st = json.loads((EXT / "fomc_statements.json").read_text())
    dates = sorted(st)
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    items = [(d, st[d], st[dates[i - 1]] if i else None) for i, d in enumerate(dates)
             if d not in done or done[d].get("n1") is None or (i and done[d].get("n2") is None)]
    print("to score:", len(items))
    with ThreadPoolExecutor(1) as ex:
        for k, r in enumerate(ex.map(score, items)):
            done[r["date"]] = r
            if k % 20 == 0:
                OUT.write_text(json.dumps(done, indent=0))
                print(k, r, flush=True)
    OUT.write_text(json.dumps(done, indent=0))
    bad = [d for d, r in done.items() if r.get("n1") is None]
    print("scored:", len(done), "missing n1:", len(bad))
