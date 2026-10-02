"""Ratio-aware resampling (see submission/RATIO_AWARE.md).

The card score is the mean over components of ours / M0, and M0's draws are reproducible at run time (m0_fallback,
same seed and cell order as the organizers' M0). For the CRPS term, the forecast that minimizes the expected ratio under
a belief p is q(y) ∝ p(y) / CRPS_M0(y). We approximate q by importance-resampling the engine's joint draws (whole rows,
so cross-asset and cross-horizon structure is kept) with weights L(x)^-gamma, where L(x) is how badly M0 would score if
the outcome were the draw x (mean over cells of CRPS_M0,cell(x) / CRPS_M0,cell(M0 median)).

gamma is 0.10 on single-cell cards and 0.25 on multi-cell cards (research/nlp/RESEARCH_LOG.md, 2026-10-03: chosen to
minimize the worse of the practice-card and pseudo-card scores). Any failure returns the draws unchanged.
"""

from __future__ import annotations

import zlib

import numpy as np

GAMMA_SINGLE = 0.10
GAMMA_MULTI = 0.25


def m0_crps_at(m0: np.ndarray, x: np.ndarray) -> np.ndarray:
    """CRPS of M0's empirical distribution (fair estimator, as the scorer) if the outcome were each row of x.
    m0: [m, d], x: [n, d] -> [n, d]."""
    m = m0.shape[0]
    srt = np.sort(m0, axis=0)
    i = np.arange(1, m + 1)[:, None]
    spread = 2.0 * np.sum((2 * i - m - 1) * srt, axis=0) / (m * (m - 1))
    out = np.empty_like(x, dtype=float)
    for k in range(x.shape[1]):
        col = srt[:, k]
        cs = np.concatenate([[0.0], np.cumsum(col)])
        pos = np.searchsorted(col, x[:, k])
        mean_abs = (x[:, k] * pos - cs[pos] + (cs[-1] - cs[pos]) - x[:, k] * (m - pos)) / m
        out[:, k] = mean_abs - 0.5 * spread[k]
    return out


def _systematic(w: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    c = np.cumsum(w / w.sum())
    return np.minimum(np.searchsorted(c, (rng.random() + np.arange(n)) / n), len(w) - 1)


def resample(samples: np.ndarray, m0_samples: np.ndarray, unit_id: str) -> tuple[np.ndarray, list[str]]:
    """samples, m0_samples: [n, n_assets, n_horizons] in the same asset/horizon order. Returns (samples, ledger)."""
    try:
        n, na, nh = samples.shape
        gamma = GAMMA_SINGLE if na * nh == 1 else GAMMA_MULTI
        if gamma <= 0:
            return samples, ["ratio-aware resampling: switched off (gamma 0)"]
        x = samples.reshape(n, -1)
        m0 = m0_samples.reshape(m0_samples.shape[0], -1)
        if m0.shape[1] != x.shape[1] or not np.isfinite(m0).all():
            return samples, ["ratio-aware resampling: M0 draws unusable, not applied"]
        loss = m0_crps_at(m0, x)
        ref = m0_crps_at(m0, np.median(m0, axis=0, keepdims=True))[0]
        L = (loss / np.maximum(ref, 1e-12)).mean(axis=1)
        w = np.maximum(L, 1e-6) ** (-gamma)
        if not np.isfinite(w).all() or w.sum() <= 0:
            return samples, ["ratio-aware resampling: degenerate weights, not applied"]
        rng = np.random.default_rng((zlib.crc32(unit_id.encode()) ^ 0x5EED) & 0x7FFFFFFF)
        idx = _systematic(w, n, rng)
        out = samples[idx]
        ess = float(w.sum() ** 2 / (w ** 2).sum())
        return out, [f"ratio-aware resampling: gamma {gamma} ({'single' if na * nh == 1 else 'multi'}-cell card), "
                     f"{len(np.unique(idx))} distinct draws kept of {n}, effective sample size {ess:.0f}"]
    except Exception as e:  # never let this cost a card
        return samples, [f"ratio-aware resampling: error {type(e).__name__}, not applied"]
