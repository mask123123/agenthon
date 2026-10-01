import itertools, numpy as np, pandas as pd
import run2
from core import make_cards, realized, m0_forecast, components, normalized
from run2 import gauss_model
wides = run2.wides
allc = make_cards(wides, start="2004-01-01", step=63)

# per-card realized |z| vs M0's own predictive sd -> used ONLY to build a shock-selected evaluation set
# (simulating how event cards were chosen), never inside a forecaster.
info = {}
for c in allc:
    w = wides[c.panel]; y = realized(c, w)
    if y is None: continue
    x = m0_forecast(c, w, 300)
    z = np.abs((y - x.mean(0)) / x.std(0)).max()
    info[c.uid] = z
zs = pd.Series(info)
thr = zs.quantile(0.67)
sets = {
    "train_all": [c for c in allc if c.uid in info and c.asof < pd.Timestamp("2015-01-01")],
    "test_all": [c for c in allc if c.uid in info and c.asof >= pd.Timestamp("2015-01-01")],
    "shock_top33": [c for c in allc if c.uid in info and info[c.uid] >= thr],
}
print({k: len(v) for k, v in sets.items()}, "shock thr z>=", round(thr, 2))

def score(model, cards):
    rows = []
    for c in cards:
        w = wides[c.panel]; y = realized(c, w); single = len(c.cells) == 1
        ref = components(m0_forecast(c, w), y, single)
        if single: ref["joint"] = 1.0
        s = normalized(components(model(c, w), y, single), ref, single)
        rows.append(min(s, 4.0))
    return float(np.mean(rows))

res = []
for drift, k in itertools.product([0.25, 0.5, 0.75, 1.0], [0.8, 0.9, 1.0, 1.1, 1.25]):
    m = gauss_model(n=300, drift=drift, k=k, nu=5)
    r = {n: score(m, cs) for n, cs in sets.items()}
    r.update(drift=drift, k=k)
    res.append(r); print({a: (round(b, 4) if isinstance(b, float) else b) for a, b in r.items()}, flush=True)
df = pd.DataFrame(res)
df["blend"] = 0.5 * (df.train_all + df.test_all) * 0.5 + 0.5 * df.shock_top33   # half uniform, half shock
df.to_csv("grid4.csv", index=False)
print(df.sort_values("blend").head(8).round(4).to_string())
