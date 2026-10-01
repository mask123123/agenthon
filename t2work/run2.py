import numpy as np, pandas as pd, zlib
from data import load_wide
from core import make_cards, history, Card, components, normalized, realized, m0_forecast
from bt import run

wides = {k: load_wide(k) for k in ["rates", "fx", "factors"]}
cards = [c for c in make_cards(wides, start="2004-01-01", step=63) if c.asof < pd.Timestamp("2015-01-01")]

def gauss_model(n=300, drift=1.0, k=1.0, nu=None, ewma=None, nd=1000):
    """M0-like engine with switches: window n, drift multiplier, width k, optional t shocks / EWMA vol."""
    def f(card, wide):
        h = history(card, wide, n)
        steps = (h.diff() if card.target_type == "level" else h.copy()).dropna()
        mu = steps.mean().values * drift
        sig = np.atleast_2d(np.cov(steps.values.T))
        if ewma:
            w = ewma ** np.arange(len(steps))[::-1]; w /= w.sum()
            x = steps.values - steps.values.mean(0)
            sig = (x * w[:, None]).T @ x
        ai = {a: i for i, a in enumerate(card.assets)}
        anchor = np.array([h[a].iloc[-1] if card.target_type == "level" else 0.0 for a in card.assets])
        cells = card.cells; d = len(cells)
        mean = np.array([anchor[ai[a]] + s * mu[ai[a]] for a, s in cells])
        cov = np.array([[min(si, sj) * sig[ai[a], ai[b]] for b, sj in cells] for a, si in cells]) * k * k
        L = np.linalg.cholesky(cov + 1e-9 * np.eye(d))
        rng = np.random.default_rng(zlib.crc32(card.uid.encode()) & 0x7FFFFFFF)
        Z = rng.standard_normal((nd, d))
        if nu:
            g = rng.chisquare(nu, size=(nd, 1)) / nu
            Z = Z / np.sqrt(g) * np.sqrt((nu - 2) / nu)
        return mean + Z @ L.T
    return f

def show(label, **kw):
    df = run(gauss_model(**kw), cards, wides)
    p = df.groupby("panel").score.mean().round(3).to_dict()
    kd = df.groupby("kind").score.mean().round(3).to_dict()
    print(f"{label:34s} all={df.score.mean():.4f} {p} {kd}", flush=True)
    return df

show("M0-like (n300, drift, gauss, 1000d)")
show("zero drift", drift=0.0)
show("zero drift k=0.9", drift=0.0, k=0.9)
show("zero drift k=0.85", drift=0.0, k=0.85)
show("zero drift t5", drift=0.0, nu=5)
show("zero drift t5 k=0.9", drift=0.0, nu=5, k=0.9)
show("zero drift n=120", drift=0.0, n=120)
show("zero drift n=600", drift=0.0, n=600)
show("zero drift n=1000", drift=0.0, n=1000)
show("zero drift ewma .97", drift=0.0, ewma=0.97)
