import sys
sys.path.insert(0, "/Users/apple/Documents/agenthon/t2work")
import numpy as np, pandas as pd, exp
df = exp.init()
def mk(**kw): return {p: dict(kw) for p in ["rates", "fx", "factors"]} | {"macro": {}}
old = exp.evaluate({p: {"beta": 0.0, "nu": 5} for p in ["rates", "fx", "factors"]} | {"macro": {}}, df)   # = previous shipped config
new = exp.evaluate({}, df)                                                                            # = new defaults
for n, o in [("previous", old), ("new defaults", new)]:
    s = exp.summarize(o)
    print(f"{n:14s}", {k: round(v, 4) for k, v in s.items()}, "overall", round(o.score.mean(), 4), o.groupby("panel").score.mean().round(3).to_dict())
d = new.score.values - old.score.values
m = old["asof"].dt.to_period("M"); g = pd.DataFrame({"m": m.values, "d": d}).groupby("m").d.agg(["sum", "count"]); months = m.unique(); rng = np.random.default_rng(2)
diffs = [(lambda a: a["sum"].sum() / a["count"].sum())(g.loc[rng.choice(months, len(months))]) for _ in range(500)]
print("new - previous:", round(d.mean(), 4), "CI", np.percentile(diffs, [2.5, 97.5]).round(4))
# by year of as-of (stability)
new["yr"] = new["asof"].dt.year; old["yr"] = old["asof"].dt.year
print((new.groupby("yr").score.mean() - old.groupby("yr").score.mean()).round(3).to_dict())
