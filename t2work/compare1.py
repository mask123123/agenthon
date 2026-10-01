import sys, json
sys.path.insert(0, "/Users/apple/Documents/agenthon/t2work")
import numpy as np, pandas as pd
import exp
df = exp.init()
best = json.load(open("best1.json"))
C0 = {"rates": {}, "fx": {}, "factors": {}, "macro": {}}
cands = {
    "C0 current": C0,
    "C1 beta.25": {p: {"beta": 0.25} for p in ["rates", "fx", "factors"]} | {"macro": {}},
    "C2 beta.25 nu8": {p: {"beta": 0.25, "nu": 8} for p in ["rates", "fx", "factors"]} | {"macro": {"nu": 8}},
    "C3 beta.25 drift.25": {p: {"beta": 0.25, "drift": 0.25} for p in ["rates", "fx", "factors"]} | {"macro": {}},
    "C4 per-panel best": best,
    "C5 beta.5": {p: {"beta": 0.5} for p in ["rates", "fx", "factors"]} | {"macro": {}},
}
res = {}
for name, P in cands.items():
    out = exp.evaluate(P, df)
    res[name] = out
    s = exp.summarize(out)
    print(f"{name:22s}", {k: round(v, 4) for k, v in s.items()}, "| overall", round(out.score.mean(), 4),
          "| by panel", out.groupby("panel").score.mean().round(3).to_dict(), flush=True)
# paired bootstrap over as-of months vs C0
base = res["C0 current"].copy(); base["m"] = base["asof"].dt.to_period("M")
months = base["m"].unique(); rng = np.random.default_rng(0)
for name, out in res.items():
    if name == "C0 current": continue
    d = (out.score.values - base.score.values)
    t = pd.DataFrame({"m": base["m"].values, "d": d, "shock": base.shock.values, "train": base.train.values})
    g = t.groupby("m").d.agg(["sum", "count"])
    diffs = []
    for _ in range(300):
        pick = rng.choice(months, len(months)); a = g.loc[pick]; diffs.append(a["sum"].sum() / a["count"].sum())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"{name:22s} mean diff vs C0 = {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
