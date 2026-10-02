"""Ratio-aware forecasting: the score is E[ours(y) / M0(y)] and M0's draws are known at run time, so the best forecast
under belief p is not p but q(y) ∝ p(y) / M0_loss(y) (CRPS is proper). Implemented by importance-resampling the
engine's joint draws (whole rows, so the joint structure is kept) with weights M0_loss(x_row)^(-gamma).

M0_loss(x) for a candidate outcome x = mean over cells of CRPS_M0,cell(x_cell) / CRPS_M0,cell(M0 median), i.e. how bad
M0 would look if x happened, normalized per cell.
"""

from __future__ import annotations

import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import review.eval_teammate as E  # noqa: E402
from backtest.evaluate import cluster_ci, components, load_all, normalized  # noqa: E402
from t2agent import m0  # noqa: E402


def m0_crps_at(m0_draws: np.ndarray, x: np.ndarray) -> np.ndarray:
    """CRPS of M0's empirical distribution if the outcome were each row of x. m0_draws [m, d], x [n, d] -> [n, d]."""
    m = m0_draws.shape[0]
    srt = np.sort(m0_draws, axis=0)
    i = np.arange(1, m + 1)[:, None]
    spread = 2.0 * np.sum((2 * i - m - 1) * srt, axis=0) / (m * (m - 1))  # E|X-X'| (fair)
    out = np.empty_like(x)
    for k in range(x.shape[1]):
        col = srt[:, k]
        cs = np.concatenate([[0.0], np.cumsum(col)])
        pos = np.searchsorted(col, x[:, k])
        # mean |X - y| via sorted cumulative sums
        mean_abs = (x[:, k] * pos - cs[pos] + (cs[-1] - cs[pos]) - x[:, k] * (m - pos)) / m
        out[:, k] = mean_abs - 0.5 * spread[k]
    return out


def systematic_resample(w: np.ndarray, n: int, rng) -> np.ndarray:
    c = np.cumsum(w / w.sum())
    u = (rng.random() + np.arange(n)) / n
    return np.minimum(np.searchsorted(c, u), len(w) - 1)


def reweight(draws: np.ndarray, m0d: np.ndarray, gamma: float, seed: int = 0) -> np.ndarray:
    if gamma == 0:
        return draws
    loss = m0_crps_at(m0d, draws)
    med = m0_crps_at(m0d, np.median(m0d, axis=0, keepdims=True))[0]
    L = (loss / np.maximum(med, 1e-12)).mean(axis=1)
    w = np.maximum(L, 1e-6) ** (-gamma)
    idx = systematic_resample(w, draws.shape[0], np.random.default_rng(seed))
    return draws[idx]


def one(args):
    card, gammas = args
    t, y = card["task"], card["y"]
    base = E.draws(t)
    m0d, _ = m0.m0_draws(t)
    out = {"id": t.unit_id, "family": t.category[-2:], "ncell": len(y), "asof": t.asof, "year": int(t.asof[:4])}
    for g in gammas:
        x = reweight(base, m0d, g)
        out[f"g{g}"] = normalized(components(x, y), card["ref"], len(y))[0]
    return out


if __name__ == "__main__":
    gammas = [0.0, 0.25, 0.5, 0.75, 1.0]
    real, pseudo = load_all()
    for lab, cards in (("real", real), ("pseudo", pseudo)):
        with ProcessPoolExecutor() as ex:
            df = pd.DataFrame(list(ex.map(one, [(c, gammas) for c in cards], chunksize=8)))
        print(f"== {lab} ({len(df)} cards) — teammate engine v2 as belief p, importance-resampled toward q ∝ p / M0_loss^gamma")
        for g in gammas:
            d = df.assign(score=df[f"g{g}"])
            lo, hi = cluster_ci(d)
            print(f"  gamma {g:<4} mean {d.score.mean():.4f} CI[{lo:.4f},{hi:.4f}]  single {d[d.ncell==1].score.mean():.4f} "
                  f"multi {d[d.ncell>1].score.mean():.4f}  early {d[d.year<2016].score.mean():.4f} late {d[d.year>=2016].score.mean():.4f}")
        df.to_csv(ROOT / "backtest" / "data" / f"ratio_aware_{lab}.csv", index=False)
