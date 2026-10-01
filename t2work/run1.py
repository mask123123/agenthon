import itertools, sys
import numpy as np, pandas as pd
from data import load_wide
from core import make_cards
from bt import run, summarize
from models import make_model

wides = {k: load_wide(k) for k in ["rates", "fx", "factors"]}
train = make_cards(wides, start="2004-01-01", step=42)
train = [c for c in train if c.asof < pd.Timestamp("2015-01-01")]
test = [c for c in make_cards(wides, start="2015-01-01", step=42)]
print("train", len(train), "test", len(test))
grid = {
    "k": [0.9, 1.0, 1.1],
    "lam": [0.94, 0.985],
    "w_long": [0.0, 0.5],
    "nu": [5, 1000],
}
res = []
for k, lam, wl, nu in itertools.product(grid["k"], grid["lam"], grid["w_long"], grid["nu"]):
    f = make_model(lam=lam, w_long=wl, k=k, nu=nu)
    df = run(f, train, wides)
    res.append(dict(k=k, lam=lam, w_long=wl, nu=nu, train=df.score.mean()))
    print(res[-1], flush=True)
r = pd.DataFrame(res).sort_values("train")
print(r.head(8).to_string())
