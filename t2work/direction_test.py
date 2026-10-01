"""Upper-bound test of DIRECTION: does the blind labeller's policy_tone (+ = tightening) predict the signed surprise of UST yields?
Outcome: z_signed = (realised level - engine mean)/engine sd for UST_2Y and UST_10Y (excluding the card's own target asset).
Then: leave-one-year-out gain in the actual CRPS-vs-M0 score from shifting the engine centre by b * tone * sd."""
import sys, tomllib
sys.path.insert(0, "/Users/apple/Documents/agenthon/t2work"); sys.path.insert(0, "/Users/apple/Documents/agenthon/submission")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from data import load_wide
from core import Card, m0_forecast, components, normalized, realized
from t2agent import engine
D = pd.read_csv("blind_analysis_data_PRIVATE.csv", parse_dates=["asof"]); D = D.drop_duplicates(["unit"])
U = "/Users/apple/Documents/agenthon/track2-forecasting-public/units"
own = {u: set(tomllib.load(open(f"{U}/{u}/card.toml", "rb"))["targets"]["asset_ids"]) for u in D.unit}
R = load_wide("rates"); rows = []
for _, r in D.iterrows():
    for a in ("UST_2Y", "UST_10Y"):
        if a in own[r.unit]: continue
        for h in (21, 63):
            c = Card(f"d-{r.unit}-{a}-{h}", "rates", r["asof"], [a], [h], "level", "single"); y = realized(c, R)
            if y is None: continue
            hist = R[a].loc[:r["asof"]].dropna()
            x = engine.build_draws({a: hist}, [a], [h], "level", c.uid, 2000).samples.reshape(-1)
            ref = components(m0_forecast(c, R), y, True); ref["joint"] = 1.0
            rows.append(dict(unit=r.unit, year=r["asof"].year, asset=a, h=h, tone=r.policy_tone, y=float(y[0]), mean=x.mean(), sd=x.std(), zs=(y[0] - x.mean()) / x.std(), x=x, ref=ref))
T = pd.DataFrame(rows); print("rows:", len(T), "units:", T.unit.nunique())
for h in (21, 63):
    t = T[T.h == h]; rho, p = spearmanr(t.tone, t.zs)
    nz = t[t.tone != 0]; hit = (np.sign(nz.tone) == np.sign(nz.zs)).mean()
    print(f"h={h:3d}: Spearman(policy_tone, signed surprise) = {rho:+.3f} (p={p:.3f}) | directional hit-rate when tone!=0: {100*hit:.0f}% (n={len(nz)}; coin-flip=50%) | share tone!=0: {100*len(nz)/len(t):.0f}%")
# economic value: shift centre by b*tone*sd, leave-one-year-out choice of b
grid = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
def score(row, b):
    x = row.x + b * row.tone * row.sd
    return min(normalized(components(x.reshape(-1, 1), np.array([row.y]), True), row.ref, True), 4.0)
S = np.array([[score(r, b) for b in grid] for r in T.itertuples()])
base = S[:, 0]; out = []
for yr in sorted(T.year.unique()):
    tr = (T.year != yr).values; te = (T.year == yr).values
    bi = S[tr].mean(0).argmin(); out.append((te.sum(), S[te, bi].mean(), base[te].mean(), grid[bi]))
n = sum(o[0] for o in out); cv = sum(o[0] * o[1] for o in out) / n; b0 = sum(o[0] * o[2] for o in out) / n
print(f"\nmean score vs M0 (lower better): engine (no text) {b0:.4f}  |  engine + direction from text, leave-one-year-out {cv:.4f}  | change {cv - b0:+.4f}")
print("in-sample score by shift size b:", dict(zip(grid, S.mean(0).round(4))))
