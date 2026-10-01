import sys, zlib
sys.path.insert(0, "/Users/apple/Documents/agenthon/submission"); sys.path.insert(0, "/Users/apple/Documents/agenthon/t2work")
import numpy as np, pandas as pd
from data import load_wide
from core import components, normalized
from t2agent import engine

W = load_wide("fx"); rng0 = np.random.default_rng(0)
idx = W.index; assets = sorted(W.columns)

def m0_hist(hist: pd.Series, h: int, uid: str):
    """M0 on a transfer history: last 300 rows, hole-dropping diffs, drift+sd, gaussian, 500 draws."""
    s = hist.iloc[-300:]
    d = s.diff(); gap = pd.Series(s.index, index=s.index).diff().dt.days
    d = d.where(gap <= max(float(gap.median()) * 10.0, 5.0)).dropna()
    r = np.random.default_rng(zlib.crc32(uid.encode()) & 0x7FFFFFFF)
    return (float(s.iloc[-1]) + h * d.mean() + np.sqrt(h) * d.std() * r.standard_normal(500))[:, None]

rows = []
asofs = idx[(idx >= "2007-01-01")][::63]
for t in asofs:
    ti = idx.get_loc(t)
    lo = idx.searchsorted(t - pd.DateOffset(years=8)); hi = idx.searchsorted(t - pd.DateOffset(years=3))
    if hi - lo < 600: continue
    for a in assets:
        early = W[a].iloc[lo:hi]; hist = pd.concat([early, W[a].iloc[ti:ti + 1]])
        ctx = [float(np.log(W[b].iloc[ti - 300:ti + 1]).diff().std()) for b in assets if b != a]
        ctx = float(np.median(ctx))
        for h in (21, 63, 126):
            if ti + h >= len(idx): continue
            y = np.array([W[a].iloc[ti + h]]); uid = f"tr-{a}-{t.date()}-{h}"
            ref = components(m0_hist(hist, h, uid), y, True); ref["joint"] = 1.0
            rec = dict(asof=t, asset=a, h=h)
            for name, kw in {"m0-engine": dict(transfer_mode="m0"), "log_early": dict(transfer_mode="log_early"), "geo": dict(transfer_mode="geo"), "ctx": dict(transfer_mode="ctx"),
                             "geo k1": dict(transfer_mode="geo", transfer_k=1.0), "geo k1.5": dict(transfer_mode="geo", transfer_k=1.5), "geo nu8": dict(transfer_mode="geo", transfer_nu=8.0)}.items():
                r = engine.build_draws({a: hist}, [a], [h], "level", uid, 1000, params=kw, context_logvol=ctx)
                rec[name] = min(normalized(components(r.samples.reshape(1000, -1), y, True), ref, True), 4.0)
            rows.append(rec)
d = pd.DataFrame(rows)
print("pseudo transfer cards:", len(d))
print(d.drop(columns=["asof", "asset", "h"]).mean().round(4).to_string())
print("by horizon:\n", d.groupby("h")[["m0-engine", "log_early", "geo", "ctx", "geo k1", "geo k1.5"]].mean().round(3).to_string())
print("early(<2015) vs late:\n", d.assign(late=d.asof >= "2015-01-01").groupby("late")[["m0-engine", "geo", "ctx"]].mean().round(3).to_string())
