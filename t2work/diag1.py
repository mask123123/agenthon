import numpy as np, pandas as pd
from data import load_wide
from core import make_cards, components, realized, m0_forecast
from run2 import gauss_model
wides = {k: load_wide(k) for k in ["factors"]}
cards = [c for c in make_cards(wides, start="2004-01-01", step=63) if c.asof < pd.Timestamp("2015-01-01") and c.family_hint == "single"]
rows = []
for c in cards:
    w = wides["factors"]; y = realized(c, w)
    if y is None: continue
    ref = components(m0_forecast(c, w), y, True)
    a = components(gauss_model(drift=1.0)(c, w), y, True)
    b = components(gauss_model(drift=0.0)(c, w), y, True)
    rows.append(dict(uid=c.uid, y=y[0], ref_m=ref["marginal"], ref_t=ref["tail"],
                     m_a=a["marginal"]/ref["marginal"], m_b=b["marginal"]/ref["marginal"],
                     t_a=a["tail"]/ref["tail"], t_b=b["tail"]/ref["tail"]))
d = pd.DataFrame(rows)
print(d[["m_a","m_b","t_a","t_b"]].describe().round(3))
print(d.sort_values("t_b", ascending=False).head(6).round(4).to_string())
