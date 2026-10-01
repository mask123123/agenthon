import sys
sys.path.insert(0, "/Users/apple/Documents/agenthon/t2work")
import numpy as np, pandas as pd, exp
df = exp.init()
multi = np.array([len(c.cells) > 1 for c in exp.CARDS]); macro = np.array([c.panel == "macro" for c in exp.CARDS])
res = {}
for n, ks in {"current (1.0)": 1.0, "k_single 0.95": 0.95, "k_single 0.9": 0.9, "k_single 0.85": 0.85, "k_single 0.8": 0.8}.items():
    o = exp.evaluate({p: {"k_single": ks} for p in ["rates", "fx", "factors"]} | {"macro": {}}, df); res[n] = o
    s = exp.summarize(o)
    print(f"{n:14s} all={o.score.mean():.4f} single(non-macro)={o[~multi & ~macro].score.mean():.4f} multi={o[multi].score.mean():.4f} | train_all {s['train_all']:.4f} test_all {s['test_all']:.4f} train_shock {s['train_shock']:.4f} test_shock {s['test_shock']:.4f}", flush=True)
base = res["current (1.0)"]; m = base["asof"].dt.to_period("M"); months = m.unique(); rng = np.random.default_rng(9)
for n, o in res.items():
    if n.startswith("current"): continue
    d = o.score.values - base.score.values
    g = pd.DataFrame({"m": m.values, "d": d}).groupby("m").d.agg(["sum", "count"])
    diffs = [(lambda a: a["sum"].sum() / a["count"].sum())(g.loc[rng.choice(months, len(months))]) for _ in range(400)]
    print(f"{n:14s} diff vs current {d.mean():+.4f} CI [{np.percentile(diffs,2.5):+.4f},{np.percentile(diffs,97.5):+.4f}]")
