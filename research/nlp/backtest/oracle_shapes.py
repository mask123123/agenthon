"""Break-even accuracy of different text-overlay SHAPES under the M0-ratio metric (analysis only, uses outcomes).

On M0's exact draws (base), a per-card signal is correct with probability p:
  shift   : move every draw by k*sd toward the signalled side
  tilt    : keep the median; stretch deviations on the signalled side by (1+g), shrink the other side by (1-g)
  mixture : move a fraction q of draws by m*sd toward the signalled side (scenario branch), rest unchanged
  calm    : signal says "quiet" -> scale width by c (<1) ; says "eventful" -> leave unchanged
            (truth: realized |y - mean| below the M0 sd on average)
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
        base, _, info = m0.m0_parts(c["task"])
        out.append((base, info["mean"], info["sd"], c["y"], c["ref"]))
    return out


def apply(shape, par, base, mean, sd, sgn, quiet, rng):
    med = np.median(base, axis=0)
    if shape == "shift":
        return base + par * sd * sgn
    if shape == "tilt":
        dev = base - med
        up = dev * sgn > 0
        return med + np.where(up, dev * (1 + par), dev * (1 - par))
    if shape == "mixture":
        q, m = par
        pick = rng.random(base.shape[0]) < q
        out = base.copy()
        out[pick] += m * sd * sgn
        return out
    if shape == "calm":
        return med + (par if quiet else 1.0) * (base - med)
    raise ValueError(shape)


def score(P, shape, par, p, seed):
    rng = np.random.default_rng(seed)
    s = []
    for base, mean, sd, y, ref in P:
        truth = np.sign(y - mean)
        truth[truth == 0] = 1
        right = rng.random() < p
        sgn = truth if right else -truth
        is_quiet = float(np.mean(np.abs(y - mean) / np.where(sd > 0, sd, 1))) < 1.0
        quiet = is_quiet if right else not is_quiet
        x = apply(shape, par, base, mean, sd, sgn, quiet, np.random.default_rng(seed + 1))
        s.append(normalized(components(x, y), ref, len(y))[0])
    return float(np.mean(s))


if __name__ == "__main__":
    real, pseudo = load_all()
    shapes = [("shift", 0.1), ("shift", 0.25), ("tilt", 0.15), ("tilt", 0.3),
              ("mixture", (0.2, 1.0)), ("mixture", (0.3, 0.75)), ("calm", 0.9), ("calm", 0.8)]
    for lab, cards in (("real", real), ("pseudo", pseudo[::2])):
        P = prep(cards)
        rows = []
        for p in (0.5, 0.55, 0.6, 0.65, 0.7, 0.8):
            r = {"p": p}
            for sh, par in shapes:
                r[f"{sh} {par}"] = np.mean([score(P, sh, par, p, s) for s in range(3)])
            rows.append(r)
        print(f"== {lab} ({len(P)} cards)")
        with pd.option_context("display.width", 250):
            print(pd.DataFrame(rows).set_index("p").round(3).to_string())
