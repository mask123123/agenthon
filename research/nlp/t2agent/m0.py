"""Exact replica of the organizer's M0 text-blind baseline (docs/M0-BASELINE.md, sections 3.1-3.9).

M0 is the denominator of every normalized score, so this module has two jobs:
  * offline: the normalizer for backtest scoring (our ratio = our component / M0 component);
  * runtime: the last-resort fallback, which by construction scores ~1.0 rather than 4.0.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from .task import Task

WINDOW = 300
N_DRAWS = 500


def _hole_threshold(dates: pd.DatetimeIndex) -> float:
    gaps = np.diff(dates.values).astype("timedelta64[D]").astype(float)
    med = float(np.median(gaps)) if len(gaps) else 1.0
    return max(10.0 * med, 5.0)


def asset_steps(s: pd.Series, target_type: str) -> pd.Series:
    """Step series of one asset over M0's trailing window, indexed by the date each step ends on."""
    w = s.iloc[-WINDOW:]
    if len(w) < 2:
        return pd.Series(dtype=float)
    thr = _hole_threshold(w.index)
    gaps = np.diff(w.index.values).astype("timedelta64[D]").astype(float)
    ok = gaps <= thr  # interval ending at rows 1..n-1
    if target_type == "log_return":
        vals = w.to_numpy(float)[1:]  # row 0 has no preceding interval
    else:
        vals = np.diff(w.to_numpy(float))
    return pd.Series(vals[ok], index=w.index[1:][ok])


def step_counts(task: Task, last_obs: dict[str, pd.Timestamp], windows: dict[str, pd.Series]) -> dict[tuple[str, int], int]:
    """Per-cell panel step count s (section 3.7). Target dates are only knowable from
    observation_periods (monthly cards); otherwise the declared horizon is used."""
    out: dict[tuple[str, int], int] = {}
    periods = task.observation_periods
    for a in task.assets:
        for k, h in enumerate(task.horizons):
            s = h
            if periods and k < len(periods):
                try:
                    td = pd.Timestamp(periods[k] + "-01" if len(periods[k]) == 7 else periods[k])
                    w = windows[a]
                    if len(w) >= 3:
                        thr = _hole_threshold(w.index)
                        gaps = np.diff(w.index.values).astype("timedelta64[D]").astype(float)
                        spacing = float(np.mean(gaps[gaps <= thr])) if (gaps <= thr).any() else 1.0
                        lo = last_obs[a]
                        if spacing > 20:
                            cnt = 12 * (td.year - lo.year) + (td.month - lo.month)
                        else:
                            cnt = int(round((td - lo).days / spacing))
                        if cnt > 0:
                            ratio = max(cnt, h) / max(min(cnt, h), 1)
                            if ratio >= 2:
                                s = cnt
                except Exception:
                    s = h
            out[(a, h)] = int(s)
    return out


def estimate(task: Task):
    """mu, Sigma (sorted asset order), anchors, per-cell step counts."""
    assets = sorted(task.assets)
    hist = {a: task.history(a) for a in assets}
    windows = {a: hist[a].iloc[-WINDOW:] for a in assets}
    steps = pd.concat({a: asset_steps(hist[a], task.target_type) for a in assets}, axis=1, join="inner").dropna()
    if len(steps) < 2:
        raise ValueError("M0: fewer than two aligned steps")
    X = steps[assets].to_numpy(float)
    mu = X.mean(axis=0)
    Sigma = np.atleast_2d(np.cov(X, rowvar=False))
    last_obs = {a: hist[a].index[-1] for a in assets}
    anchor = {a: (0.0 if task.target_type == "log_return" else float(hist[a].iloc[-1])) for a in assets}
    s = step_counts(task, last_obs, windows)
    return assets, mu, Sigma, anchor, s


def m0_draws(task: Task, n_draws: int = N_DRAWS) -> tuple[np.ndarray, list[tuple[str, int]]]:
    draws, cells, _ = m0_parts(task, n_draws)
    return draws, cells


def m0_parts(task: Task, n_draws: int = N_DRAWS) -> tuple[np.ndarray, list[tuple[str, int]], dict]:
    """M0's draws plus its per-cell mean, sd and step counts (for common-random-number engines)."""
    assets, mu, Sigma, anchor, s = estimate(task)
    idx = {a: i for i, a in enumerate(assets)}
    cells = task.cells()
    d = len(cells)
    mean = np.array([anchor[a] + s[(a, h)] * mu[idx[a]] for a, h in cells])
    cov = np.empty((d, d))
    for i, (ai, hi) in enumerate(cells):
        for j, (aj, hj) in enumerate(cells):
            cov[i, j] = min(s[(ai, hi)], s[(aj, hj)]) * Sigma[idx[ai], idx[aj]]
    cov[np.diag_indices(d)] += 1e-10
    try:
        L = np.linalg.cholesky(cov + 1e-9 * np.eye(d))
    except np.linalg.LinAlgError:
        L = np.diag(np.sqrt(np.clip(np.diag(cov + 1e-9 * np.eye(d)), 0, None)))
    seed = zlib.crc32(task.unit_id.encode()) & 0x7FFFFFFF
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_draws, d))
    info = {"mean": mean, "sd": np.sqrt(np.diag(cov)), "steps": s, "mu": dict(zip(assets, mu)), "anchor": anchor}
    return mean + Z @ L.T, cells, info
