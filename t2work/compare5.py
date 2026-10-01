import sys
sys.path.insert(0, "/Users/apple/Documents/agenthon/t2work")
import numpy as np, pandas as pd, exp
from core import components
df = exp.init()
kinds = np.array([c.family_hint for c in exp.CARDS]); multi = np.array([len(c.cells) > 1 for c in exp.CARDS])
print("multi-cell cards:", int(multi.sum()), pd.Series(kinds[multi]).value_counts().to_dict())
def mk(**kw): return {p: dict(kw) for p in ["rates", "fx", "factors"]} | {"macro": {}}
M0L = dict(drift=1.0, beta=0.0, nu=1000.0, k=1.0, k_family={})
cands = {
  "CUR": mk(),
  "M0-like": mk(**M0L),
  "drift1 only": mk(drift=1.0),
  "drift1 + beta.5": mk(drift=1.0, nu=1000.0),
  "drift1 + nu8": mk(drift=1.0, beta=0.0),
  "drift.75": mk(drift=0.75),
  "drift1 beta.25 nu8": mk(drift=1.0, beta=0.25),
}
res = {}
for n, P in cands.items():
    o = exp.evaluate(P, df); res[n] = o
    o2 = o.copy(); o2["kind"] = kinds; o2["multi"] = multi
    print(f"{n:20s} all={o.score.mean():.4f} single={o2[~multi].score.mean():.4f} multi={o2[multi].score.mean():.4f}  multi by kind", o2[multi].groupby('kind').score.mean().round(3).to_dict(), flush=True)
base = res["CUR"]; m = base["asof"].dt.to_period("M"); months = m.unique(); rng = np.random.default_rng(5)
for n, o in res.items():
    if n == "CUR": continue
    for lab, mask in [("multi", multi), ("all", np.ones(len(multi), bool))]:
        d = o.score.values[mask] - base.score.values[mask]
        g = pd.DataFrame({"m": m.values[mask], "d": d}).groupby("m").d.agg(["sum", "count"]); mo = g.index.values
        diffs = [(lambda a: a["sum"].sum() / a["count"].sum())(g.loc[rng.choice(mo, len(mo))]) for _ in range(300)]
        print(f"{n:20s} {lab:5s} diff vs CUR {d.mean():+.4f} CI [{np.percentile(diffs,2.5):+.4f},{np.percentile(diffs,97.5):+.4f}]")
