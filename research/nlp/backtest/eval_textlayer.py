"""Report the v1 Nemotron run: reliability, cost, signal frequency, and the score effect of the frozen
overlay on the teammate engine (base) for the cards that were run."""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import review.eval_teammate as E  # noqa: E402
from backtest.evaluate import components, load_all, normalized  # noqa: E402
from t2agent.textlayer import overlay  # noqa: E402

RUN = ROOT / "backtest" / "data" / "nemotron_v1"

if __name__ == "__main__":
    recs = [json.loads(p.read_text()) for p in sorted(RUN.glob("*.json"))]
    secs = np.array([r["seconds"] for r in recs])
    reqs = np.array([r["requests"] for r in recs])
    toks = [sum((e.get("usage") or {}).get("completion_tokens", 0) for e in r["log"]) for r in recs]
    fails = sum(not e.get("ok") for r in recs for e in r["log"])
    print(f"cards {len(recs)}  ok {sum(r['ok'] for r in recs)}  failed calls {fails}")
    print(f"seconds p50 {np.median(secs):.0f}  p95 {np.percentile(secs, 95):.0f}  max {secs.max():.0f}")
    print(f"requests p50 {np.median(reqs):.0f} max {reqs.max()}   output tokens p50 {np.median(toks):.0f} max {max(toks)}")
    red = [e for r in recs for e in r["log"] if e.get("thinking")]
    print("reduce finish reasons:", pd.Series([e.get("finish") for e in red]).value_counts().to_dict(),
          " reasoning tokens p50/max:", int(np.median([(e.get("usage") or {}).get("completion_tokens_details", {}).get("reasoning_tokens", 0) for e in red])),
          max((e.get("usage") or {}).get("completion_tokens_details", {}).get("reasoning_tokens", 0) for e in red))

    real, _ = load_all()
    by = {c["task"].unit_id: c for c in real}
    rows, freq = [], []
    for r in recs:
        c = by[r["uid"]]
        t, y = c["task"], c["y"]
        base = E.draws(t)
        card = (r["result"] or {}).get("card")
        new, ledger = overlay(base, t.cells(), card)
        s0 = normalized(components(base, y), c["ref"], len(y))[0]
        s1 = normalized(components(new, y), c["ref"], len(y))[0]
        rows.append({"uid": r["uid"], "family": t.category[-2:], "base": s0, "overlay": s1, "changed": s1 != s0,
                     "ledger": "; ".join(ledger)[:160]})
        for a, v in ((card or {}).get("assets") or {}).items():
            if isinstance(v, dict):
                act = v.get("established_action") or {}
                freq.append({"dir": act.get("direction", 0), "conf": act.get("confidence", 0.5),
                             "event": (v.get("binary_event_in_horizon") or {}).get("present") is True,
                             "stress": v.get("stress"), "regime": (v.get("regime_constraint") or {}).get("type")})
    df, fq = pd.DataFrame(rows), pd.DataFrame(freq)
    print("\nsignal frequency over", len(fq), "asset entries:")
    print("  established action (dir!=0):", int((fq["dir"] != 0).sum()), " of which conf>=0.7:",
          int(((fq["dir"] != 0) & (pd.to_numeric(fq["conf"], errors="coerce") >= 0.7)).sum()))
    print("  binary event in horizon:", int(fq["event"].sum()), "  stress:", fq["stress"].value_counts().to_dict())
    print("  regime constraint:", fq["regime"].value_counts().to_dict())
    print(f"\nscore on these {len(df)} cards: base {df.base.mean():.4f}  overlay {df.overlay.mean():.4f}  "
          f"delta {df.overlay.mean() - df.base.mean():+.4f}  cards changed {int(df.changed.sum())}")
    print(df[df.changed][["uid", "family", "base", "overlay", "ledger"]].to_string(index=False))
