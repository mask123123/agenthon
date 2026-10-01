import sys
sys.path.insert(0, "/Users/apple/Documents/agenthon/t2work")
import numpy as np, pandas as pd, exp
df = exp.init()
multi = np.array([len(c.cells) > 1 for c in exp.CARDS])
def mk(**kw): return {p: dict(kw) for p in ["rates", "fx", "factors"]} | {"macro": {}}
cands = {"CUR": mk(), "corr_n 600": mk(corr_n=600), "corr_n 1500": mk(corr_n=1500), "shrink .25": mk(corr_shrink=0.25),
         "shrink .5": mk(corr_shrink=0.5), "corr_n 600 sh.25": mk(corr_n=600, corr_shrink=0.25), "corr_n 150": mk(corr_n=150)}
res = {}
for n, P in cands.items():
    o = exp.evaluate(P, df); res[n] = o
    mm = o[multi]; print(f"{n:18s} multi-cell n={len(mm)} mean={mm.score.mean():.4f}  train={mm[mm.train].score.mean():.4f} test={mm[~mm.train].score.mean():.4f} shock={mm[mm.shock].score.mean():.4f} | by panel", mm.groupby('panel').score.mean().round(3).to_dict(), flush=True)
base = res["CUR"][multi]; m = base["asof"].dt.to_period("M"); months = m.unique(); rng = np.random.default_rng(4)
for n, o in res.items():
    if n == "CUR": continue
    d = o[multi].score.values - base.score.values
    g = pd.DataFrame({"m": m.values, "d": d}).groupby("m").d.agg(["sum", "count"])
    diffs = [(lambda a: a["sum"].sum() / a["count"].sum())(g.loc[rng.choice(months, len(months))]) for _ in range(300)]
    print(f"{n:18s} diff vs CUR (multi-cell) {d.mean():+.4f} CI [{np.percentile(diffs,2.5):+.4f},{np.percentile(diffs,97.5):+.4f}]")
