"""Exact replica of the organizer's M0 baseline (track docs/M0-BASELINE.md sections 3.1-3.9), used as the fallback
when the main engine fails. M0 is the score's denominator, so this fallback scores ~1.0 on a card instead of the
4.0 a failed card costs. The first 500 draws are M0's own draws (same seed, same cell order); extra draws continue
the same random stream.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

WINDOW = 300


def _gaps(idx: pd.DatetimeIndex) -> np.ndarray:
    return np.diff(idx.values).astype("timedelta64[D]").astype(float)


def _threshold(idx: pd.DatetimeIndex) -> float:
    g = _gaps(idx)
    return max(10.0 * float(np.median(g)) if len(g) else 10.0, 5.0)


def _window(s: pd.Series) -> pd.Series:
    s = pd.Series(np.asarray(s.values, dtype=float), index=pd.to_datetime(pd.Index(s.index).astype(str).str[:10]))
    s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
    return s.iloc[-WINDOW:]


def _steps(w: pd.Series, target_type: str) -> pd.Series:
    if len(w) < 2:
        return pd.Series(dtype=float)
    ok = _gaps(w.index) <= _threshold(w.index)
    vals = w.to_numpy()[1:] if target_type == "log_return" else np.diff(w.to_numpy())
    return pd.Series(vals[ok], index=w.index[1:][ok])


def _step_count(w: pd.Series, h: int, period: str | None) -> int:
    """Declared horizon, unless an explicit target month disagrees with it by 2x or more (section 3.7)."""
    if not period or len(w) < 3:
        return int(h)
    try:
        td = pd.Timestamp(period + "-01" if len(period) == 7 else period)
        g = _gaps(w.index)
        ok = g[g <= _threshold(w.index)]
        spacing = float(ok.mean()) if len(ok) else 1.0
        lo = w.index[-1]
        cnt = 12 * (td.year - lo.year) + (td.month - lo.month) if spacing > 20 else int(round((td - lo).days / spacing))
        if cnt <= 0:
            return int(h)
        return cnt if max(cnt, h) / max(min(cnt, h), 1) >= 2 else int(h)
    except Exception:
        return int(h)


def m0_draws(hist: dict, assets: list[str], horizons: list[int], target_type: str, unit_id: str, n_draws: int,
             observation_periods: list[str] | None = None) -> np.ndarray:
    """Returns samples [n_draws, n_assets, n_horizons] in the card's asset/horizon order. Raises if M0 cannot run."""
    sa = sorted(assets)
    win = {a: _window(hist[a]) for a in sa}
    steps = pd.concat({a: _steps(win[a], target_type) for a in sa}, axis=1, join="inner").dropna()
    if len(steps) < 2:
        raise ValueError("M0 fallback: fewer than two aligned steps")
    X = steps[sa].to_numpy(float)
    mu, sigma = X.mean(axis=0), np.atleast_2d(np.cov(X, rowvar=False))
    idx = {a: i for i, a in enumerate(sa)}
    period_of = dict(zip(horizons, observation_periods)) if observation_periods and len(observation_periods) == len(horizons) else {}
    cells = [(a, h) for a in sa for h in sorted(horizons)]
    s = {(a, h): _step_count(win[a], h, period_of.get(h)) for a, h in cells}
    anchor = {a: 0.0 if target_type == "log_return" else float(win[a].iloc[-1]) for a in sa}
    d = len(cells)
    mean = np.array([anchor[a] + s[(a, h)] * mu[idx[a]] for a, h in cells])
    cov = np.array([[min(s[ci], s[cj]) * sigma[idx[ci[0]], idx[cj[0]]] for cj in cells] for ci in cells])
    cov[np.diag_indices(d)] += 1e-10
    try:
        L = np.linalg.cholesky(cov + 1e-9 * np.eye(d))
    except np.linalg.LinAlgError:
        L = np.diag(np.sqrt(np.clip(np.diag(cov + 1e-9 * np.eye(d)), 0, None)))
    rng = np.random.default_rng(zlib.crc32(unit_id.encode()) & 0x7FFFFFFF)
    flat = mean + rng.standard_normal((max(n_draws, 500), d)) @ L.T
    flat = flat[:n_draws] if n_draws >= 500 else flat[:500]
    out = np.empty((flat.shape[0], len(assets), len(horizons)))
    pos = {c: k for k, c in enumerate(cells)}
    for ai, a in enumerate(assets):
        for hi, h in enumerate(horizons):
            out[:, ai, hi] = flat[:, pos[(a, h)]]
    if not np.isfinite(out).all():
        raise ValueError("M0 fallback: non-finite draws")
    return out
