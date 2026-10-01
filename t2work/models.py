"""Executive summary: candidate forecasters. All use only data at/before the card's as-of.
Engine = zero (or shrunk) drift random walk, EWMA/long-window blended daily vol, window correlation,
multivariate-t shocks with a common mixing variable, one global width multiplier k."""
from __future__ import annotations
import zlib
import numpy as np
from core import Card, history


def _ewma_vol(x: np.ndarray, lam: float) -> np.ndarray:
    w = lam ** np.arange(len(x))[::-1]
    w /= w.sum()
    return np.sqrt((w[:, None] * x ** 2).sum(0))


def make_model(lam=0.97, w_long=0.3, long_n=1500, corr_n=750, k=1.0, nu=6.0, drift_shrink=0.0,
               fat_extra=1.0, n_draws=1000):
    def f(card: Card, wide) -> np.ndarray:
        hfull = history(card, wide)
        steps_all = (hfull.diff() if card.target_type == "level" else hfull.copy()).dropna()
        st_long = steps_all.iloc[-long_n:]
        st_corr = steps_all.iloc[-corr_n:]
        # centered vols (ignore sample mean for vol)
        v_short = _ewma_vol(st_corr.values - 0.0, lam)
        v_long = np.sqrt((st_long.values ** 2).mean(0))
        vol = np.sqrt((1 - w_long) * v_short ** 2 + w_long * v_long ** 2)
        R = np.atleast_2d(np.corrcoef(st_corr.values.T))
        ai = {a: i for i, a in enumerate(card.assets)}
        anchor = np.array([hfull[a].iloc[-1] if card.target_type == "level" else 0.0 for a in card.assets])
        mu_s = st_corr.mean().values * drift_shrink
        cells = card.cells
        d = len(cells)
        mean = np.array([anchor[ai[a]] + s * mu_s[ai[a]] for a, s in cells])
        cov = np.empty((d, d))
        for i, (a, si) in enumerate(cells):
            for j, (b, sj) in enumerate(cells):
                cov[i, j] = min(si, sj) * vol[ai[a]] * vol[ai[b]] * R[ai[a], ai[b]]
        cov = cov * k * k + 1e-12 * np.eye(d)
        try:
            L = np.linalg.cholesky(cov + 1e-12 * np.eye(d))
        except np.linalg.LinAlgError:
            L = np.diag(np.sqrt(np.diag(cov) + 1e-12))
        rng = np.random.default_rng(zlib.crc32(card.uid.encode()) & 0x7FFFFFFF)
        Z = rng.standard_normal((n_draws, d))
        if nu and nu < 1e3:                         # multivariate t, unit-variance scaling
            g = rng.chisquare(nu, size=(n_draws, 1)) / nu
            Z = Z / np.sqrt(g) * np.sqrt((nu - 2) / nu)
        return mean + fat_extra * (Z @ L.T)
    return f
