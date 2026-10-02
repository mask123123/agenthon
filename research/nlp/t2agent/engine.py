"""Statistical forecasting engine (text-blind). Every component is a config knob so it can be ablated.

Model, per card:
  * steps: level -> first differences, log_return -> log(1 + r); steps spanning a data hole dropped.
  * vol term structure: per-step variance starts at a recent (EWMA) estimate and decays toward a
    long-run estimate with half-life `vol_mr_hl`, so long horizons lean on the long-run level.
  * dependence: one correlation matrix (recent aligned steps, shrunk to identity) drives every
    asset; horizons share one path (h2 = h1 + independent continuation), as the variogram wants.
  * tails: a per-draw volatility multiplier v = exp(tau*g - tau^2) (E[v^2] = 1), common across
    assets and horizons - a regime-uncertainty scale mixture that fattens horizon-level tails.
  * drift: M0's trailing-window mean step, shrunk by `drift_shrink` (level) / long-run mean of
    log(1+r), shrunk by `lr_drift_shrink` (log_return).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import m0
from .task import Task


@dataclass
class Config:
    n_draws: int = 4000
    ewma_hl: float = 20.0  # half-life (steps) of the recent variance estimate
    long_window: int = 1260  # steps in the long-run variance estimate (~5y daily)
    w_recent: float = 1.0  # weight of EWMA in the starting variance (rest: long-run)
    vol_mr_hl: float = 60.0  # half-life (steps) of decay from starting variance to long-run
    corr_window: int = 250
    corr_shrink: float = 0.1
    tau: float = 0.0  # scale-mixture log-vol dispersion (0 = Gaussian)
    width: float = 1.0  # global sd multiplier
    family_width: dict = field(default_factory=dict)  # e.g. {"T2-F4": 1.2}
    drift_shrink: float = 0.0  # level: fraction of M0's trailing mean step kept
    lr_drift_shrink: float = 0.0  # log_return: fraction of long-run mean log-return kept
    lr_drift_m0: bool = False  # log_return: use M0's trailing mean step (times drift_shrink) instead
    seed: int = 0
    # common random numbers: transform M0's exact draws instead of sampling afresh
    crn: bool = False
    crn_vol: bool = False  # use our vol model for the per-cell sd (else keep M0's sd)
    sd_ratio_bounds: tuple = (0.5, 2.0)


def _steps(s: pd.Series, target_type: str) -> pd.Series:
    if len(s) < 2:
        return pd.Series(dtype=float)
    gaps = np.diff(s.index.values).astype("timedelta64[D]").astype(float)
    thr = max(10.0 * float(np.median(gaps[-300:])), 5.0)
    ok = gaps <= thr
    if target_type == "log_return":
        vals = np.log1p(np.clip(s.to_numpy(float)[1:], -0.99, None))
    else:
        vals = np.diff(s.to_numpy(float))
    return pd.Series(vals[ok], index=s.index[1:][ok])


def _ewma_var(x: np.ndarray, hl: float) -> float:
    lam = 0.5 ** (1.0 / hl)
    w = lam ** np.arange(len(x))[::-1]
    return float(np.sum(w * x**2) / np.sum(w))


def forecast(task: Task, cfg: Config) -> tuple[np.ndarray, list[tuple[str, int]]]:
    assets = sorted(task.assets)
    hist = {a: task.history(a) for a in assets}
    steps = {a: _steps(hist[a], task.target_type) for a in assets}
    for a in assets:
        if len(steps[a]) < 10:
            raise ValueError(f"engine: too little history for {a}")

    # per-step variance term structure, per asset
    var_now, var_lr, drift = {}, {}, {}
    _, mu_m0, _, anchor, s_cnt = m0.estimate(task)
    for i, a in enumerate(assets):
        x = steps[a].to_numpy(float)
        xc = x - x[-m0.WINDOW:].mean() if task.target_type == "log_return" else x
        lr = float(np.var(xc[-cfg.long_window:]))
        rec = _ewma_var(xc[-max(int(cfg.ewma_hl * 8), 50):], cfg.ewma_hl)
        var_lr[a] = lr
        var_now[a] = cfg.w_recent * rec + (1 - cfg.w_recent) * lr
        if task.target_type == "log_return" and not cfg.lr_drift_m0:
            drift[a] = cfg.lr_drift_shrink * float(x[-cfg.long_window * 4:].mean())
        else:
            drift[a] = cfg.drift_shrink * float(mu_m0[i])

    phi = 0.5 ** (1.0 / cfg.vol_mr_hl)

    def cumvar(a: str, s: int) -> float:
        if s <= 0:
            return 0.0
        decay = phi * (1 - phi**s) / (1 - phi)
        return s * var_lr[a] + (var_now[a] - var_lr[a]) * decay

    # correlation of recent aligned steps
    if len(assets) > 1:
        al = pd.concat(steps, axis=1, join="inner").dropna().iloc[-cfg.corr_window:]
        R = np.corrcoef(al[assets].to_numpy(float), rowvar=False) if len(al) > 20 else np.eye(len(assets))
        R = np.nan_to_num(R, nan=0.0)
        R = (1 - cfg.corr_shrink) * R + cfg.corr_shrink * np.eye(len(assets))
        np.fill_diagonal(R, 1.0)
        w, V = np.linalg.eigh(R)
        R = (V * np.clip(w, 1e-6, None)) @ V.T
        d = np.sqrt(np.diag(R))
        R = R / np.outer(d, d)
        Lr = np.linalg.cholesky(R)
    else:
        Lr = np.ones((1, 1))

    mult = cfg.width * float(cfg.family_width.get(task.category, 1.0))
    seed = (zlib.crc32(task.unit_id.encode()) ^ cfg.seed) & 0x7FFFFFFF
    rng = np.random.default_rng(seed)

    if cfg.crn:
        base, cells, info = m0.m0_parts(task)
        n = base.shape[0]
        v = np.exp(cfg.tau * rng.standard_normal(n) - cfg.tau**2) if cfg.tau > 0 else np.ones(n)
        sd_ours = np.array([np.sqrt(max(cumvar(a, s_cnt[(a, h)]), 0.0)) for a, h in cells])
        ratio = np.where(info["sd"] > 0, sd_ours / np.where(info["sd"] > 0, info["sd"], 1.0), 1.0)
        d = np.clip(ratio, *cfg.sd_ratio_bounds) if cfg.crn_vol else np.ones(len(cells))
        centre = np.array([anchor[a] + s_cnt[(a, h)] * drift[a] for a, h in cells])
        return centre + mult * v[:, None] * d[None, :] * (base - info["mean"][None, :]), cells

    n = cfg.n_draws
    v = np.exp(cfg.tau * rng.standard_normal(n) - cfg.tau**2) if cfg.tau > 0 else np.ones(n)

    # path over the union of step counts: segments between successive distinct step counts
    cells = task.cells()
    per_asset_s = {a: sorted({s_cnt[(a, h)] for h in task.horizons}) for a in assets}
    # common segment grid across assets (step counts can differ per asset on monthly cards)
    grid = sorted({s for a in assets for s in per_asset_s[a]})
    level = {a: np.zeros(n) for a in assets}
    at = {}
    prev = 0
    for s in grid:
        Z = rng.standard_normal((n, len(assets))) @ Lr.T
        for i, a in enumerate(assets):
            seg_var = max(cumvar(a, s) - cumvar(a, prev), 0.0)
            level[a] = level[a] + np.sqrt(seg_var) * Z[:, i]
            at[(a, s)] = level[a].copy()
        prev = s
    out = np.empty((n, len(cells)))
    for k, (a, h) in enumerate(cells):
        s = s_cnt[(a, h)]
        out[:, k] = anchor[a] + s * drift[a] + mult * v * at[(a, s)]
    return out, cells
