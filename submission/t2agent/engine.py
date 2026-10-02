"""Executive summary (read this first)

`build_draws` turns a card's panels into joint Monte-Carlo draws. Method (validated by our own
walk-forward backtest against the published M0 spec, see t2work/): last-300-step window of steps
(level -> first differences with hole-dropping, log_return -> log(1+r)), drift = DRIFT * sample mean,
covariance = sd_a sd_b R_ab * min(s_i, s_j) so draws are paths across assets and horizons, and
multivariate-t(NU) shocks with one shared mixing variable. Everything uses data at/before the as-of.
No text is read in this version.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

DEFAULTS = {
    "window": 300,        # trailing steps for drift/vol/corr
    "drift": 0.5,         # multiplier on sample mean step
    "nu": 8.0,            # Student-t dof (shared mixing variable)
    "k": 1.0,             # global width multiplier
    "k_family": {"T2-F4": 1.10, "T2-F2": 1.05},   # a-priori: event-type families (card family is an allowed input)
    "w_long": 0.0,        # variance weight on the full-history vol
    "beta": 0.5,          # vol-of-vol: sd *= (sd_short / sd_window) ** beta
    "short_n": 21,        # steps in the short vol window
    "half_life": 0,       # >0: vol term structure, current vol decays to long-run with this half-life (steps)
    "ewma_lam": 0.94,     # EWMA decay for the *current* variance in the term-structure model
    "corr_n": 0,          # >0: estimate correlation on this many trailing steps (default: the vol window)
    "corr_shrink": 0.0,   # shrink correlation toward the constant (average) correlation
    "drift_n": 0,         # >0: estimate drift on this many trailing steps instead of the vol window
    "hurst": 0.5,         # horizon scaling exponent (0.5 = random walk)
    "transfer_k": 1.0, "transfer_nu": 4.0,
    "k_single": 1.0,       # extra width multiplier for single-cell cards only (non-monthly)
    "mix_p": 0.0, "mix_m": 1.0,   # shock mixture: with prob mix_p a draw has its whole deviation scaled by mix_m (shared across cells)
    "robust": "", "m_robust": "",   # "" | "mad" | "winsor3" | "winsor4": outlier-robust step sd (daily / monthly panels)
    "transfer_mode": "geo",   # m0 | log_early | geo (early log-vol x context log-vol, geometric mean) | ctx
    "monthly_drift": 1.0, "monthly_k": 1.0, "monthly_nu": 5.0,
}
# Multi-cell cards (several assets and/or horizons) are scored with a variogram that is a SQUARED error
# normalised by M0's own error, so any departure from M0's drift/covariance structure is punished hard.
# Backtest (t2work/compare5.py, search2.py): single-cell params give 1.034 on multi-cell cards vs ~0.997 here.
MULTI = {"drift": 0.9, "nu": 8.0, "k": 1.05, "beta": 0.0}
# Optional build-time override (see Dockerfile ARG CONFIG_JSON): lets one code base ship width-calibration variants.
import json as _json, os as _os
_cfg = _os.path.join(_os.path.dirname(__file__), "config.json")
if _os.path.exists(_cfg):
    try:
        _o = _json.load(open(_cfg))
        for _k, _v in _o.items():
            if _k in DEFAULTS:
                DEFAULTS[_k] = _v
    except Exception:
        pass
WINDOW = DEFAULTS["window"]


@dataclass
class Result:
    samples: np.ndarray                      # [n_draws, n_assets, n_horizons]
    meta: dict[str, Any] = field(default_factory=dict)


def _hole_mask(s: pd.Series) -> pd.Series:
    """True where the step from the previous observation spans a data hole."""
    when = pd.to_datetime(pd.Series(s.index, index=s.index), errors="coerce")
    gap = when.diff().dt.days
    if gap.notna().sum() == 0:
        return pd.Series(False, index=s.index)
    return gap > max(float(gap.median()) * 10.0, 5.0)


def _steps(s: pd.Series, target_type: str, monthly: bool, window: int) -> tuple[pd.Series, bool]:
    """ALL steps (full history) plus a flag: did the recent window contain a hole (transfer card)?"""
    s = s.astype(float)
    if target_type == "log_return":
        r = np.asarray(s.values, dtype=float)
        ok = np.isfinite(r) & (r > -1.0)
        st = pd.Series(np.where(ok, np.log1p(np.where(ok, r, 0.0)), np.nan), index=s.index)
        return st.dropna(), False
    d = s.diff()
    if monthly:
        per = pd.PeriodIndex(pd.to_datetime(s.index), freq="M")
        consec = np.r_[False, np.diff(per.asi8) == 1]
        return d.where(consec).dropna(), False
    holes = _hole_mask(s.iloc[-window:])
    # hole-spanning steps are dropped on the whole history, using the window's own spacing rule
    all_holes = _hole_mask(s)
    return d.where(~all_holes).dropna(), bool(holes.iloc[1:].any())


def _psd_corr(c: np.ndarray) -> np.ndarray:
    c = np.nan_to_num(c, nan=0.0)
    np.fill_diagonal(c, 1.0)
    w, v = np.linalg.eigh((c + c.T) / 2)
    c = v @ np.diag(np.clip(w, 1e-6, None)) @ v.T
    dg = np.sqrt(np.diag(c))
    return c / np.outer(dg, dg)


def _spacing_days(s: pd.Series, window: int) -> float:
    """Mean calendar-day spacing of the asset's trailing observations, ignoring data holes."""
    idx = pd.to_datetime(pd.Series(s.index[-(window + 1):]))
    gap = idx.diff().dt.days.dropna()
    if len(gap) < 3:
        return 1.4
    ok = gap[gap <= max(float(gap.median()) * 10.0, 5.0)]
    return float(ok.mean()) if len(ok) else 1.4


def _panel_steps_for(h: int, spacing: float) -> int:
    """Declared business-day horizon -> panel steps. Same 2x rule as M0 (docs/M0-BASELINE.md 3.7):
    keep the declared horizon unless counting steps from calendar spacing disagrees by >= 2x."""
    counted = max(int(round(h * 7.0 / 5.0 / spacing)), 1)
    ratio = max(counted, h) / max(min(counted, h), 1)
    return counted if ratio >= 2.0 else int(h)


def _robust_sd(x: pd.Series, mode: str) -> float:
    """Step-volatility estimate that is not dominated by a single extreme month/day (e.g. April 2020 unemployment)."""
    v = np.asarray(x.values, dtype=float)
    if not mode or len(v) < 20:
        return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med))
    if mad <= 0:
        return float(np.std(v, ddof=1))
    if mode == "mad":
        return float(mad)
    c = 3.0 if mode == "winsor3" else 4.0
    return float(np.std(np.clip(v, med - c * mad, med + c * mad), ddof=1))


def _fbm_cov(si: float, sj: float, h: float) -> float:
    """Covariance of a fractional walk at two horizons; h=0.5 gives min(si, sj)."""
    if abs(h - 0.5) < 1e-12:
        return min(si, sj)
    return 0.5 * (si ** (2 * h) + sj ** (2 * h) - abs(si - sj) ** (2 * h))


def build_draws(hist: dict[str, pd.Series], assets: list[str], horizons: list[int], target_type: str,
                unit_id: str, n_draws: int, family: str = "", panel_steps: np.ndarray | None = None,
                params: dict | None = None, context_logvol: float | None = None) -> Result:
    monthly = panel_steps is not None
    P = dict(DEFAULTS)
    if len(assets) * len(horizons) > 1 and not monthly:
        P.update(MULTI)
    P.update(params or {})
    W = int(P["window"])
    steps, transfer = {}, {}
    for a in assets:
        st, tr = _steps(hist[a], target_type, monthly, W)
        steps[a], transfer[a] = st, tr
    if any(len(steps[a]) < 20 for a in assets):
        raise ValueError("not enough history for at least one asset")

    win = {a: steps[a].iloc[-W:] for a in assets}
    rmode = P["m_robust"] if monthly else P["robust"]
    sd_win = np.array([_robust_sd(win[a], rmode) for a in assets])
    if P["w_long"] > 0:
        sd_long = np.array([steps[a].std() for a in assets])
        sd = np.sqrt((1 - P["w_long"]) * sd_win ** 2 + P["w_long"] * sd_long ** 2)
    else:
        sd = sd_win.copy()
    if P["beta"] and not monthly:                 # monthly cards keep the plain window vol
        sn = int(P["short_n"])
        sd21 = np.array([np.sqrt((win[a].iloc[-sn:] ** 2).mean()) for a in assets])
        sd = sd * np.clip(sd21 / sd_win, 0.5, 2.0) ** P["beta"]
    dn = int(P["drift_n"]) if P["drift_n"] else W
    mu = np.array([steps[a].iloc[-dn:].mean() for a in assets])
    # Transfer assets (target history = early window + hole + as-of row, level targets): level differences from
    # the early window are in the wrong scale (and may sit in a crisis). Re-estimate step vol from LOG returns
    # (scale-free, whole early window), optionally blended with the vol of other assets at the as-of, then
    # re-express in level units at the as-of level. Early-window drift is meaningless -> 0.
    if target_type == "level" and not monthly and P["transfer_mode"] != "m0":
        for i, a in enumerate(assets):
            if not transfer[a]:
                continue
            lv = hist[a].astype(float)
            lv = lv[lv > 0]
            lr = np.log(lv).diff()
            lr = lr.where(~_hole_mask(lv)).dropna()
            if len(lr) < 100:
                continue
            s_early = float(lr.std())
            mode = P["transfer_mode"]
            if mode == "ctx" and context_logvol:
                s_use = float(context_logvol)
            elif mode == "geo" and context_logvol:
                s_use = float(np.sqrt(s_early * context_logvol))
            else:
                s_use = s_early
            sd[i] = float(lv.iloc[-1]) * s_use
            mu[i] = 0.0
    if not np.isfinite(sd).all() or (sd <= 0).any():
        raise ValueError("degenerate volatility")
    cn = int(P["corr_n"]) if P["corr_n"] else W
    frame = pd.DataFrame({a: steps[a].iloc[-cn:] for a in assets})
    corr = _psd_corr(frame.corr(min_periods=30).to_numpy()) if len(assets) > 1 else np.ones((1, 1))
    if len(assets) > 1 and P["corr_shrink"]:
        off = corr[~np.eye(len(assets), dtype=bool)].mean()
        tgt = np.full_like(corr, off); np.fill_diagonal(tgt, 1.0)
        corr = _psd_corr((1 - P["corr_shrink"]) * corr + P["corr_shrink"] * tgt)

    last = np.array([0.0 if target_type == "log_return" else float(hist[a].iloc[-1]) for a in assets])
    spacing = {a: _spacing_days(hist[a], W) for a in assets}
    cells = [(ai, hi, (panel_steps[ai, hi] if monthly else _panel_steps_for(horizons[hi], spacing[assets[ai]])))
             for ai in range(len(assets)) for hi in range(len(horizons))]
    d = len(cells)
    any_transfer = any(transfer.values())
    drift = P["monthly_drift"] if monthly else P["drift"]
    k = (P["monthly_k"] if monthly else P["k"]) * P["k_family"].get(family, 1.0) * (P["transfer_k"] if any_transfer else 1.0)
    if len(assets) * len(horizons) == 1 and not monthly:
        k *= P["k_single"]
    nu = P["transfer_nu"] if any_transfer else (P["monthly_nu"] if monthly else P["nu"])

    mean = np.array([last[ai] + drift * mu[ai] * s for ai, _, s in cells], dtype=float)
    cov = np.empty((d, d))
    if P["half_life"] and not monthly:
        hmax = int(max(c[2] for c in cells))
        rho = 0.5 ** (1.0 / float(P["half_life"]))
        lam = float(P["ewma_lam"])
        v_inf = np.array([win[a].var() for a in assets])
        if P["w_long"] > 0:
            v_inf = (1 - P["w_long"]) * v_inf + P["w_long"] * np.array([steps[a].var() for a in assets])
        v0 = []
        for a in assets:
            x = win[a].values - win[a].values.mean()
            wts = lam ** np.arange(len(x))[::-1]; wts /= wts.sum()
            v0.append(float((wts * x ** 2).sum()))
        v0 = np.clip(np.array(v0), 0.25 * v_inf, 4.0 * v_inf)
        t = np.arange(hmax)
        sig = np.sqrt(np.maximum(v_inf[:, None] + (v0 - v_inf)[:, None] * rho ** t[None, :], 1e-18))   # [A, hmax]
        cum = {}
        for ai in range(len(assets)):
            for bj in range(len(assets)):
                cum[(ai, bj)] = np.cumsum(sig[ai] * sig[bj])
        for i, (ai, _, si) in enumerate(cells):
            for j, (bj, _, sj) in enumerate(cells):
                cov[i, j] = cum[(ai, bj)][int(min(si, sj)) - 1] * corr[ai, bj]
    else:
        for i, (ai, _, si) in enumerate(cells):
            for j, (bj, _, sj) in enumerate(cells):
                cov[i, j] = _fbm_cov(si, sj, P["hurst"]) * sd[ai] * sd[bj] * corr[ai, bj]
    cov = cov * k * k
    scale = np.diag(cov).mean() or 1.0
    try:
        L = np.linalg.cholesky(cov + 1e-9 * scale * np.eye(d))
    except np.linalg.LinAlgError:
        L = np.diag(np.sqrt(np.diag(cov) + 1e-12))
    rng = np.random.default_rng(zlib.crc32(unit_id.encode()) & 0x7FFFFFFF)
    z = rng.standard_normal((n_draws, d))
    g = rng.chisquare(nu, size=(n_draws, 1)) / nu
    z = z / np.sqrt(g) * np.sqrt((nu - 2.0) / nu)
    dev = z @ L.T
    if P["mix_p"] > 0 and P["mix_m"] != 1.0:
        shock = rng.random((n_draws, 1)) < P["mix_p"]
        dev = dev * np.where(shock, P["mix_m"], 1.0)
    flat = mean + dev
    if not np.isfinite(flat).all():
        raise ValueError("non-finite draws")
    out = np.empty((n_draws, len(assets), len(horizons)))
    for c, (ai, hi, _) in enumerate(cells):
        out[:, ai, hi] = flat[:, c]
    meta = {"k": k, "nu": nu, "drift_mult": drift, "transfer": any_transfer, "monthly": monthly,
            "params": {x: P[x] for x in ("window", "w_long", "beta", "hurst")},
            "n_steps": {a: int(len(steps[a])) for a in assets}, "spacing_days": spacing, "cell_steps": [int(c[2]) for c in cells], "last": dict(zip(assets, last.tolist())),
            "step_sd": dict(zip(assets, sd.tolist())), "step_mean": dict(zip(assets, mu.tolist()))}
    return Result(out, meta)


def fallback_draws(hist: dict[str, pd.Series], assets: list[str], horizons: list[int], target_type: str,
                   unit_id: str, n_draws: int, panel_steps: np.ndarray | None = None) -> Result:
    """Last-resort Gaussian walk. Uses whatever history exists; never raises.

    Horizons are converted to panel steps exactly like the main engine (explicit monthly steps when given, otherwise
    the M0 spacing rule), so a monthly macro card no longer uses the business-day key as a step count."""
    rng = np.random.default_rng(zlib.crc32(unit_id.encode()) & 0x7FFFFFFF)
    out = np.empty((n_draws, len(assets), len(horizons)))
    info = {}
    for ai, a in enumerate(assets):
        s = hist.get(a)
        v = np.asarray(s.values, dtype=float) if s is not None and len(s) else np.array([0.0])
        v = v[np.isfinite(v)] if np.isfinite(v).any() else np.array([0.0])
        if target_type == "log_return":
            st = np.log1p(np.clip(v, -0.99, None)); anchor = 0.0
        else:
            st = np.diff(v) if len(v) > 2 else np.array([0.0]); anchor = float(v[-1])
        sd_ = float(np.std(st[-300:])) if len(st) > 2 and np.std(st[-300:]) > 0 else max(abs(anchor) * 0.01, 1e-4)
        try:
            spacing = _spacing_days(s, WINDOW) if s is not None and len(s) > 3 else 1.4
        except Exception:
            spacing = 1.4
        for hi, h in enumerate(horizons):
            try:
                steps = int(panel_steps[ai, hi]) if panel_steps is not None else _panel_steps_for(int(h), spacing)
            except Exception:
                steps = int(h)
            out[:, ai, hi] = anchor + rng.standard_normal(n_draws) * sd_ * np.sqrt(max(steps, 1)) * 1.1
        info[a] = {"anchor": anchor, "step_sd": sd_, "spacing_days": spacing}
    return Result(out, {"fallback": True, "info": info})
