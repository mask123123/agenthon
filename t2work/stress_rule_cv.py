"""Does the (contaminated, negative) stress signal improve actual forecast SCORES out of sample?
Per card-date: pseudo single-cell cards on reference assets (excluding the card's own target) at h=21,63; score = normalised vs M0.
Rule family: width multiplier m(stress) chosen per stress level on TRAIN years only, evaluated on the HELD-OUT year (leave-one-year-out).
Compare with: global best multiplier (also chosen on train years) and with the unchanged engine (m=1)."""
import sys, json, tomllib
sys.path.insert(0, "/Users/apple/Documents/agenthon/t2work"); sys.path.insert(0, "/Users/apple/Documents/agenthon/submission")
import numpy as np, pandas as pd
from data import load_wide
from core import Card, m0_forecast, components, normalized, realized
from t2agent import engine
D = pd.read_csv("blind_analysis_data_PRIVATE.csv", parse_dates=["asof"])
U = "/Users/apple/Documents/agenthon/track2-forecasting-public/units"
own = {u: set(tomllib.load(open(f"{U}/{u}/card.toml", "rb"))["targets"]["asset_ids"]) for u in D.unit.unique()}
W = {k: load_wide(k) for k in ["rates", "fx", "factors"]}
REF = {"UST_2Y": ("rates", "level"), "UST_10Y": ("rates", "level"), "EUR": ("fx", "level"), "JPY": ("fx", "level"), "GBP": ("fx", "level"), "MKT": ("factors", "log_return")}
MULT = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35]
rows = []
for (unit, asof), g in D.groupby(["unit", "asof"]):
    st = float(g.stress.iloc[0])
    for h in (21, 63):
        sc = {m: [] for m in MULT}
        for a, (pn, tt) in REF.items():
            if a in own[unit]: continue
            c = Card(f"s-{unit}-{a}-{h}", pn, asof, [a], [h], tt, "single"); w = W[pn]
            y = realized(c, w)
            if y is None: continue
            ref = components(m0_forecast(c, w), y, True); ref["joint"] = 1.0
            hist = w[a].loc[:asof].dropna()
            for m in MULT:
                try:
                    x = engine.build_draws({a: hist}, [a], [h], tt, c.uid, 1000, params={"k": m, "k_family": {}}).samples.reshape(1000, -1)
                    sc[m].append(min(normalized(components(x, y, True), ref, True), 4.0))
                except Exception:
                    sc[m].append(4.0)
        if sc[1.0]:
            rows.append(dict(unit=unit, asof=asof, year=asof.year, stress=st, h=h, **{f"m{m}": float(np.mean(v)) for m, v in sc.items()}))
S = pd.DataFrame(rows); S.to_csv("stress_rule_scores_PRIVATE.csv", index=False)
print("card-date-horizon rows:", len(S), "| stress levels:", S.stress.value_counts().sort_index().to_dict())
cols = [f"m{m}" for m in MULT]
print("\nmean normalised score (vs M0=1.0) by stress level and width multiplier (all rows, descriptive):")
print(S.groupby("stress")[cols].mean().round(3).to_string())
# leave-one-year-out
res = {"engine (m=1)": [], "global best m": [], "per-stress best m": []}
for yr in sorted(S.year.unique()):
    tr, te = S[S.year != yr], S[S.year == yr]
    g = tr[cols].mean().idxmin()
    best = {s: tr[tr.stress == s][cols].mean().idxmin() for s in tr.stress.unique()}
    for i, r in te.iterrows():
        res["engine (m=1)"].append(r["m1.0"]); res["global best m"].append(r[g]); res["per-stress best m"].append(r[best.get(r.stress, "m1.0")])
print("\nLEAVE-ONE-YEAR-OUT mean score:", {k: round(float(np.mean(v)), 4) for k, v in res.items()})
a, b = np.array(res["per-stress best m"]), np.array(res["engine (m=1)"]); d = a - b
rng = np.random.default_rng(0); bs = [np.mean(rng.choice(d, len(d))) for _ in range(2000)]
print("per-stress rule minus engine: %+.4f   bootstrap 95%% CI [%+.4f, %+.4f]" % (d.mean(), *np.percentile(bs, [2.5, 97.5])))
