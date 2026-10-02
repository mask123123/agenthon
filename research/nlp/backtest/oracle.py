"""What is a directional / magnitude signal worth under the M0-ratio metric?

Analysis only (uses realized outcomes by design). On top of M0's exact draws (common random
numbers), shift each cell's centre by k * sd in a signalled direction that is correct with
probability p; separately, scale the width by (1+g) when a signal says "big move" and (1-g)
otherwise, correct with probability p. Prints the mean normalized score per (p, k).
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from backtest.evaluate import components, load_all, normalized  # noqa: E402
from t2agent import m0  # noqa: E402


def prep(cards):
    out = []
    for c in cards:
        base, cells, info = m0.m0_parts(c["task"])
        out.append((base, info["mean"], info["sd"], c["y"], c["ref"], c["task"].category[-2:]))
    return out


def score_dir(P, p, k, rng):
    sc = []
    for base, mean, sd, y, ref, fam in P:
        truth = np.sign(y - mean)
        truth[truth == 0] = 1
        ok = rng.random() < p  # one call per card: the signal is right or wrong for the whole card
        sgn = truth if ok else -truth
        x = base + (k * sd * sgn)[None, :]
        sc.append(normalized(components(x, y), ref, len(y))[0])
    return float(np.mean(sc))


def score_width(P, p, g, rng):
    sc = []
    for base, mean, sd, y, ref, fam in P:
        big = float(np.mean(np.abs(y - mean) / np.where(sd > 0, sd, 1))) > 1.0
        ok = rng.random() < p
        say_big = big if ok else not big
        f = 1 + g if say_big else 1 - g
        x = mean[None, :] + f * (base - mean[None, :])
        sc.append(normalized(components(x, y), ref, len(y))[0])
    return float(np.mean(sc))


if __name__ == "__main__":
    real, pseudo = load_all()
    for lab, cards in (("real", real), ("pseudo", pseudo[::2])):
        P = prep(cards)
        rows = []
        for p in (0.5, 0.6, 0.7, 0.8, 1.0):
            r = {"p": p}
            for k in (0.1, 0.25, 0.5):
                r[f"dir k={k}"] = np.mean([score_dir(P, p, k, np.random.default_rng(s)) for s in range(3)])
            for g in (0.15, 0.3):
                r[f"width g={g}"] = np.mean([score_width(P, p, g, np.random.default_rng(s)) for s in range(3)])
            rows.append(r)
        print(f"== {lab} ({len(P)} cards)")
        print(pd.DataFrame(rows).set_index("p").round(3).to_string())
