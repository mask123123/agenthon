import itertools, numpy as np, pandas as pd
from data import load_wide
from core import make_cards
from bt import run
import run2
from run2 import gauss_model

wides = run2.wides
train = [c for c in make_cards(wides, start="2004-01-01", step=63) if c.asof < pd.Timestamp("2015-01-01")]
test = [c for c in make_cards(wides, start="2015-01-01", step=63)]
res = []
for n, drift, k, nu in itertools.product([200, 300, 450], [0.75, 1.0, 1.25], [0.85, 0.92, 1.0], [None, 5]):
    df = run(gauss_model(n=n, drift=drift, k=k, nu=nu), train, wides)
    p = df.groupby("panel").score.mean()
    res.append(dict(n=n, drift=drift, k=k, nu=nu, all=df.score.mean(), rates=p["rates"], fx=p["fx"], factors=p["factors"]))
    print(res[-1], flush=True)
r = pd.DataFrame(res).sort_values("all")
r.to_csv("grid3_train.csv", index=False)
print(r.head(10).round(4).to_string())
best = r.iloc[0]
dft = run(gauss_model(n=int(best.n), drift=best.drift, k=best.k, nu=None if pd.isna(best.nu) else int(best.nu)), test, wides)
print("TEST best-of-train:", round(dft.score.mean(), 4), dft.groupby("panel").score.mean().round(3).to_dict())
