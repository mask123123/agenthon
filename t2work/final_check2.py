import sys
sys.path.insert(0, "/Users/apple/Documents/agenthon/t2work")
import numpy as np, pandas as pd, exp
df = exp.init()
kinds = np.array([c.family_hint for c in exp.CARDS]); multi = np.array([len(c.cells) > 1 for c in exp.CARDS])
single_only = {p: {"drift": 0.5, "beta": 0.5, "nu": 8.0, "k": 1.0} for p in ["rates", "fx", "factors"]}   # previous shipped (all shapes)
new = exp.evaluate({}, df)
old = exp.evaluate({p: {**v, "k_family": {}} for p, v in single_only.items()} | {"macro": {}}, df)
for n, o in [("previous", old), ("new (shape-split)", new)]:
    o2 = o.copy(); o2["multi"] = multi; o2["kind"] = kinds
    s = exp.summarize(o)
    print(f"{n:18s} all={o.score.mean():.4f} single={o2[~multi].score.mean():.4f} multi={o2[multi].score.mean():.4f} | train_all {s['train_all']:.4f} test_all {s['test_all']:.4f} test_shock {s['test_shock']:.4f}")
    print("   by kind:", o2.groupby("kind").score.mean().round(3).to_dict(), "| by panel:", o2.groupby("panel").score.mean().round(3).to_dict())
d = new.score.values - old.score.values
m = old["asof"].dt.to_period("M"); g = pd.DataFrame({"m": m.values, "d": d}).groupby("m").d.agg(["sum", "count"]); mo = g.index.values; rng = np.random.default_rng(6)
diffs = [(lambda a: a["sum"].sum() / a["count"].sum())(g.loc[rng.choice(mo, len(mo))]) for _ in range(500)]
print("new - previous (all cards):", round(d.mean(), 4), "CI", np.percentile(diffs, [2.5, 97.5]).round(4))
# realistic roster mix: 58% single, 42% multi
w = 0.58 * new[~multi].score.mean() + 0.42 * new[multi].score.mean()
w0 = 0.58 * old[~multi].score.mean() + 0.42 * old[multi].score.mean()
print("roster-weighted (58/42):  previous", round(w0, 4), " new", round(w, 4))
