"""Small, capped centre shift for UST cells from Fed tone (research: research/nlp/RESEARCH_LOG.md).

For each UST target asset: x = [sign of M0's drift, n1, n2] -> P(outcome above M0's centre) by a frozen logistic
model (coefs.json) -> shift = clip(0.25 * (2P - 1), -0.10, 0.10) x sd of that cell's draws. Non-UST cells are never
touched. Any failure (no endpoint, no statement, unparseable answer, budget, timeout) returns the draws unchanged.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from .fed_tone import find_statements, tone
from .house import House

_COEFS = json.loads((pathlib.Path(__file__).parent / "coefs.json").read_text())


def _m0_drift_sign(series) -> float:
    v = np.asarray(series, dtype=float)[-301:]
    v = v[np.isfinite(v)]
    return float(np.sign(np.diff(v).mean())) if len(v) > 2 else 0.0


def _prob(x: list[float]) -> float:
    z = (np.asarray(x) - np.asarray(_COEFS["mean"])) / np.asarray(_COEFS["scale"])
    return float(1.0 / (1.0 + np.exp(-(_COEFS["intercept"] + float(np.dot(_COEFS["weights"], z))))))


def text_overlay(samples: np.ndarray, assets: list[str], horizons: list[int], hist: dict, text_dir, asof: str,
                 house: House | None = None) -> tuple[np.ndarray, list[str]]:
    """samples: [n_draws, n_assets, n_horizons] (the teammate engine's layout). Returns (samples, ledger)."""
    try:
        ust = [i for i, a in enumerate(assets) if str(a).startswith("UST_") and a in hist]
        if not ust:
            return samples, ["text layer: no UST targets, not applied"]
        statements = find_statements(pathlib.Path(text_dir), asof)
        if not statements:
            return samples, ["text layer: no FOMC statement in the corpus, not applied"]
        house = house or House()
        if not house.available:
            return samples, ["text layer: no model endpoint, not applied"]
        latest = statements[-1]
        previous = statements[-2][1] if len(statements) > 1 else None
        n1, n2 = tone(house, latest[1], previous)
        if n1 is None:
            return samples, [f"text layer: no usable tone reading ({house.used} requests), not applied"]
        n2_used = n2 if n2 is not None else _COEFS["mean"][2]  # missing comparison -> neutral (training mean)
        out = samples.copy()
        ledger = [f"text layer: FOMC statement {latest[0]} hawkishness n1={n1:.2f} (1-5), "
                  f"change vs previous n2={'n/a' if n2 is None else f'{n2:+.2f}'}; {house.used} House requests"]
        cap, k = _COEFS["mapping"]["cap_sd"], _COEFS["mapping"]["shift_sd_per_unit"]
        for ai in ust:
            a = assets[ai]
            trend = _m0_drift_sign(getattr(hist[a], "values", hist[a]))
            p = _prob([trend, n1, n2_used])
            shift = float(np.clip(k * (2 * p - 1), -cap, cap))
            for hi in range(len(horizons)):
                col = out[:, ai, hi]
                out[:, ai, hi] = col + shift * float(np.std(col))
            ledger.append(f"  {a}: P(above M0 centre)={p:.2f} -> centre shift {shift:+.3f} sd (cap {cap})")
        if not np.isfinite(out).all():
            return samples, ["text layer: non-finite result, not applied"]
        return out, ledger
    except Exception as e:  # never let the text layer cost a card
        return samples, [f"text layer: error {type(e).__name__}, not applied"]
