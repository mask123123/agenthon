import sys, itertools
sys.path.insert(0, "/Users/apple/Documents/agenthon/t2work")
import numpy as np, pandas as pd, exp
df = exp.init()
multi = np.array([len(c.cells) > 1 and c.panel != "macro" for c in exp.CARDS])
rows = []
for drift, nu, k, beta in itertools.product([0.9, 1.0, 1.1], [6, 8, 12], [0.95, 1.0, 1.05], [0.0, 0.15]):
    P = {p: dict(drift=drift, nu=float(nu), k=k, beta=beta, k_family={}) for p in ["rates", "fx", "factors"]}
    o = exp.evaluate(P, df, only=multi)
    tr, te = o.train, ~o.train
    rows.append(dict(drift=drift, nu=nu, k=k, beta=beta, train=o[tr].score.mean(), test=o[te].score.mean(), shock=o[o.shock].score.mean(), all=o.score.mean()))
    print({a: (round(b, 4) if isinstance(b, float) else b) for a, b in rows[-1].items()}, flush=True)
r = pd.DataFrame(rows).sort_values("train")
r.to_csv("search2.csv", index=False)
print(r.head(8).round(4).to_string())
