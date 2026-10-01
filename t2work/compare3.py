import sys
sys.path.insert(0, "/Users/apple/Documents/agenthon/t2work")
import numpy as np, pandas as pd, exp
df = exp.init()
def mk(**kw): return {p: dict(kw) for p in ["rates", "fx", "factors"]} | {"macro": {}}
cands = {
  "CUR (beta.5)": mk(),
  "TS hl10": mk(half_life=10, beta=0), "TS hl30": mk(half_life=30, beta=0), "TS hl90": mk(half_life=90, beta=0),
  "TS hl30 lam.97": mk(half_life=30, beta=0, ewma_lam=0.97),
  "TS hl30 wl.25": mk(half_life=30, beta=0, w_long=0.25),
  "drift_n 60": mk(drift_n=60), "drift_n 120": mk(drift_n=120), "drift_n 600": mk(drift_n=600),
  "drift_n 120 x.75": mk(drift_n=120, drift=0.75), "drift_n 600 x.75": mk(drift_n=600, drift=0.75), "drift_n 600 x1": mk(drift_n=600, drift=1.0),
}
res = {}
for n, P in cands.items():
    o = exp.evaluate(P, df); res[n] = o; s = exp.summarize(o)
    print(f"{n:18s}", {k: round(v, 4) for k, v in s.items()}, "| all", round(o.score.mean(), 4), o.groupby("panel").score.mean().round(3).to_dict(), flush=True)
base = res["CUR (beta.5)"]; m = base["asof"].dt.to_period("M"); months = m.unique(); rng = np.random.default_rng(3)
for n, o in res.items():
    if n.startswith("CUR"): continue
    d = o.score.values - base.score.values
    g = pd.DataFrame({"m": m.values, "d": d}).groupby("m").d.agg(["sum", "count"])
    diffs = [(lambda a: a["sum"].sum() / a["count"].sum())(g.loc[rng.choice(months, len(months))]) for _ in range(300)]
    print(f"{n:18s} diff vs CUR {d.mean():+.4f} CI [{np.percentile(diffs,2.5):+.4f},{np.percentile(diffs,97.5):+.4f}]")
