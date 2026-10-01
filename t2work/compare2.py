import sys, json
sys.path.insert(0, "/Users/apple/Documents/agenthon/t2work")
import numpy as np, pandas as pd
import exp
df = exp.init()
def mk(**kw): return {p: dict(kw) for p in ["rates", "fx", "factors"]} | {"macro": {}}
cands = {"C0": mk(), "b.5 s21": mk(beta=0.5), "b.75 s21": mk(beta=0.75), "b1.0 s21": mk(beta=1.0),
         "b.5 s10": mk(beta=0.5, short_n=10), "b.5 s42": mk(beta=0.5, short_n=42), "b.75 s42": mk(beta=0.75, short_n=42),
         "b.5 s21 nu8": mk(beta=0.5, nu=8), "b.5 s21 k1.05": mk(beta=0.5, k=1.05), "b.5 s21 k.95": mk(beta=0.5, k=0.95)}
res = {}
for name, P in cands.items():
    out = exp.evaluate(P, df); res[name] = out; s = exp.summarize(out)
    print(f"{name:14s}", {k: round(v, 4) for k, v in s.items()}, "| overall", round(out.score.mean(), 4), flush=True)
base = res["C0"]; base_m = base["asof"].dt.to_period("M"); months = base_m.unique(); rng = np.random.default_rng(1)
for name, out in res.items():
    if name == "C0": continue
    d = out.score.values - base.score.values
    g = pd.DataFrame({"m": base_m.values, "d": d}).groupby("m").d.agg(["sum", "count"])
    diffs = [(lambda a: a["sum"].sum() / a["count"].sum())(g.loc[rng.choice(months, len(months))]) for _ in range(300)]
    print(f"{name:14s} diff vs C0 {d.mean():+.4f}  CI [{np.percentile(diffs,2.5):+.4f}, {np.percentile(diffs,97.5):+.4f}]")
