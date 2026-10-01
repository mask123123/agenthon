import sys
sys.path.insert(0, "/Users/apple/Documents/agenthon/t2work")
import numpy as np, pandas as pd, exp
from core import components
exp.init()
rows = []
for i, c in enumerate(exp.CARDS):
    if len(c.cells) == 1 or c.panel == "macro": continue
    y, ref = exp.Y[i], exp.REF[i]
    for name, P in {"CUR": {}, "nodrift": {"drift": 0.0}, "drift1": {"drift": 1.0}, "beta0": {"beta": 0.0}, "nu1000": {"nu": 1000}}.items():
        x = exp._forecast(c, P)
        comp = components(x, y, False)
        rows.append(dict(panel=c.panel, kind=c.family_hint, variant=name, m=comp["marginal"] / ref["marginal"], j=comp["joint"] / ref["joint"], t=comp["tail"] / ref["tail"]))
d = pd.DataFrame(rows)
print(d.groupby(["panel", "kind", "variant"])[["m", "j", "t"]].mean().round(3).to_string())
