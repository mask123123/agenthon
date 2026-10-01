import sys
sys.path.insert(0, "/Users/apple/Documents/agenthon/t2work")
import numpy as np, pandas as pd, exp
df = exp.init()
macro = np.array([c.panel == "macro" for c in exp.CARDS])
res = {}
for n, P in {"std (current)": {}, "mad": {"m_robust": "mad"}, "winsor3": {"m_robust": "winsor3"}, "winsor4": {"m_robust": "winsor4"},
             "mad k1.1": {"m_robust": "mad", "monthly_k": 1.1}, "mad k0.9": {"m_robust": "mad", "monthly_k": 0.9},
             "winsor3 d.5": {"m_robust": "winsor3", "monthly_drift": 0.5}, "winsor3 nu8": {"m_robust": "winsor3", "monthly_nu": 8.0}}.items():
    o = exp.evaluate({"macro": P}, df, panels=["macro"]); res[n] = o
    sh = o.shock
    print(f"{n:16s} macro n={len(o)} mean={o.score.mean():.4f} train={o[o.train].score.mean():.4f} test={o[~o.train].score.mean():.4f} shock={o[sh].score.mean():.4f}", flush=True)
base = res["std (current)"]; m = base["asof"].dt.to_period("M"); months = m.unique(); rng = np.random.default_rng(7)
for n, o in res.items():
    if n.startswith("std"): continue
    d = o.score.values - base.score.values
    g = pd.DataFrame({"m": m.values, "d": d}).groupby("m").d.agg(["sum", "count"])
    diffs = [(lambda a: a["sum"].sum() / a["count"].sum())(g.loc[rng.choice(months, len(months))]) for _ in range(300)]
    print(f"{n:16s} diff vs std {d.mean():+.4f} CI [{np.percentile(diffs,2.5):+.4f},{np.percentile(diffs,97.5):+.4f}]")
