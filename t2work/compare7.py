"""GARCH(1,1) volatility forecast and AR(1) centre vs our current engine, on SINGLE-cell cards (where we gain)."""
import sys, zlib
sys.path.insert(0, "/Users/apple/Documents/agenthon/t2work"); sys.path.insert(0, "/Users/apple/Documents/agenthon/submission")
import numpy as np, pandas as pd
from scipy.optimize import minimize
from multiprocessing import get_context
import exp
from core import components, normalized
from t2agent import engine
df = exp.init()

def garch_fit(x):
    x = x - x.mean(); v0 = x.var()
    def nll(p):
        w, a, b = p
        if w <= 0 or a < 0 or b < 0 or a + b >= 0.999: return 1e10
        s2 = np.empty(len(x)); s2[0] = v0
        for t in range(1, len(x)): s2[t] = w + a * x[t-1] ** 2 + b * s2[t-1]
        return 0.5 * np.sum(np.log(s2) + x ** 2 / s2)
    best = None
    for a0, b0 in [(0.05, 0.9), (0.1, 0.8)]:
        r = minimize(nll, [v0 * (1 - a0 - b0), a0, b0], method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-9, "maxiter": 300})
        if best is None or r.fun < best.fun: best = r
    w, a, b = best.x
    s2 = v0
    for t in range(1, len(x)): s2 = w + a * x[t-1] ** 2 + b * s2
    s2_next = w + a * x[-1] ** 2 + b * s2          # variance of step t+1
    return w, a, b, s2_next

def alt_forecast(card, w_, mode):
    hist = w_[card.assets[0]].loc[:card.asof].dropna()
    tt = card.target_type
    s = hist.astype(float)
    st = (np.log1p(s) if tt == "log_return" else s.diff()).dropna()
    h = card.horizons[0]; win = st.iloc[-300:].values; x1000 = st.iloc[-1000:].values
    mu = win.mean() * 0.5
    last = 0.0 if tt == "log_return" else float(s.iloc[-1])
    var_tot = win.var(ddof=1) * h
    centre_shift = 0.0
    if mode in ("garch", "garch+ar"):
        wq, a, b, s2n = garch_fit(x1000)
        pers = a + b; vinf = wq / max(1 - pers, 1e-6)
        j = np.arange(h)
        var_tot = float(np.sum(vinf + pers ** j * (s2n - vinf)))
    if mode in ("ar", "garch+ar"):
        d = win - win.mean(); phi = float(np.sum(d[1:] * d[:-1]) / np.sum(d[:-1] ** 2)); phi = float(np.clip(phi, -0.3, 0.3))
        centre_shift = phi * (st.iloc[-1] - win.mean()) * (1 - phi ** h) / (1 - phi) if abs(phi) > 1e-9 else 0.0
        var_tot *= ((1 + phi) / (1 - phi)) if abs(phi) > 1e-9 else 1.0
    rng = np.random.default_rng(zlib.crc32(card.uid.encode()) & 0x7FFFFFFF)
    nu = 8.0
    z = rng.standard_normal(1000) / np.sqrt(rng.chisquare(nu, 1000) / nu) * np.sqrt((nu - 2) / nu)
    return (last + mu * h + centre_shift + np.sqrt(var_tot) * z)[:, None]

def score_one(args):
    i, mode = args
    c = exp.CARDS[i]
    try:
        x = exp._forecast(c, {}) if mode == "CUR" else alt_forecast(c, exp.WIDES[c.panel], mode)
        s = normalized(components(x, exp.Y[i], True), exp.REF[i], True)
    except Exception:
        s = 4.0
    return min(s, 4.0)

single = [i for i, c in enumerate(exp.CARDS) if len(c.cells) == 1 and c.panel != "macro"]
rng = np.random.default_rng(0)
sub = sorted(rng.choice(single, size=1600, replace=False).tolist())      # GARCH MLE is slow: random 1600 single-cell cards
res = {}
for mode in ["CUR", "garch", "ar", "garch+ar"]:
    with get_context("fork").Pool(8) as p:
        res[mode] = np.array(p.map(score_one, [(i, mode) for i in sub], chunksize=20))
    tr = np.array([df["train"].iloc[i] for i in sub]); sh = np.array([df["shock"].iloc[i] for i in sub])
    print(f"{mode:9s} n={len(sub)} mean={res[mode].mean():.4f} train={res[mode][tr].mean():.4f} test={res[mode][~tr].mean():.4f} shock={res[mode][sh].mean():.4f}", flush=True)
months = pd.Series([df["asof"].iloc[i] for i in sub]).dt.to_period("M"); mo = months.unique(); rg = np.random.default_rng(1)
for mode in ["garch", "ar", "garch+ar"]:
    d = res[mode] - res["CUR"]
    g = pd.DataFrame({"m": months.values, "d": d}).groupby("m").d.agg(["sum", "count"])
    diffs = [(lambda a: a["sum"].sum() / a["count"].sum())(g.loc[rg.choice(mo, len(mo))]) for _ in range(300)]
    print(f"{mode:9s} diff vs CUR {d.mean():+.4f} CI [{np.percentile(diffs,2.5):+.4f},{np.percentile(diffs,97.5):+.4f}]")
