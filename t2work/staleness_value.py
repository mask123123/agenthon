"""How much is ONE extra month of macro data worth? Pseudo-cards on the public monthly macro panel.
Stale (what M0 and the panel-only engine see): history through month t-1, target s+1 steps ahead.
Fresh (what a parsed release gives us): history through month t, target s steps ahead. Score = fresh engine vs M0-stale."""
import sys
sys.path.insert(0, "/Users/apple/Documents/agenthon/t2work"); sys.path.insert(0, "/Users/apple/Documents/agenthon/submission")
import numpy as np, pandas as pd
from data import load_wide
from core import Card, m0_forecast, components, normalized
from t2agent import engine
W = load_wide("macro"); idx = W.index; assets = ["CPI_ALL", "CPI_CORE", "NFP", "UNRATE", "PCE_ALL", "PCE_CORE"]
rows = []
for i in range(120, len(idx) - 10):
    for a in assets:
        for s in (2, 3, 8):
            if i + s >= len(idx): continue
            y = np.array([float(W[a].iloc[i + s])])
            stale_card = Card(f"st-{a}-{i}-{s}", "macro", idx[i - 1], [a], [s + 1], "level", "single")
            ref = components(m0_forecast(stale_card, W), y, True); ref["joint"] = 1.0
            out = {}
            for name, (end, steps) in {"fresh": (i, s), "stale_engine": (i - 1, s + 1)}.items():
                hist = W[a].iloc[:end + 1].dropna()
                x = engine.build_draws({a: hist}, [a], [s], "level", f"u-{a}-{i}-{s}-{name}", 1000, panel_steps=np.array([[steps]], float)).samples.reshape(1000, -1)
                out[name] = min(normalized(components(x, y, True), ref, True), 4.0)
            rows.append(dict(asset=a, s=s, year=idx[i].year, **out))
D = pd.DataFrame(rows)
print("rows:", len(D))
print("mean score vs M0(stale):  stale-engine %.3f | FRESH (with the newest print) %.3f" % (D.stale_engine.mean(), D.fresh.mean()))
print("\nby asset (fresh / stale-engine):"); print(D.groupby("asset")[["fresh", "stale_engine"]].mean().round(3).to_string())
print("\nby horizon steps s (fresh / stale-engine):"); print(D.groupby("s")[["fresh", "stale_engine"]].mean().round(3).to_string())
d = (D.fresh - D.stale_engine).values; rng = np.random.default_rng(0); bs = [np.mean(rng.choice(d, len(d))) for _ in range(1000)]
print("\nfresh minus stale-engine: %+.4f  95%% CI [%+.4f, %+.4f]" % (d.mean(), *np.percentile(bs, [2.5, 97.5])))
