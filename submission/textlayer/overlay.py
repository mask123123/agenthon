"""Small, capped centre shift for UST cells from Fed tone (research: research/nlp/RESEARCH_LOG.md).

For each UST target asset: x = [sign of M0's drift, tone level, tone change vs previous statement] -> P(outcome above
M0's centre) by a frozen logistic model -> shift = clip(0.25 * (2P - 1), -0.10, 0.10) x sd of that cell's draws.
Tone comes from the House model when it answers (model "house"), else from the frozen phrase lexicon (model
"lexicon", no network). Sentiment readings are added to the ledger only. Non-UST cells are never touched; width is
never changed; any unexpected error returns the draws unchanged.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from . import lexicon
from .fed_tone import find_statements, tone
from .house import House

_COEFS = json.loads((pathlib.Path(__file__).parent / "coefs.json").read_text())


def _m0_drift_sign(series) -> float:
    v = np.asarray(series, dtype=float)[-301:]
    v = v[np.isfinite(v)]
    return float(np.sign(np.diff(v).mean())) if len(v) > 2 else 0.0


def _prob(model: dict, x: list[float]) -> float:
    z = (np.asarray(x) - np.asarray(model["mean"])) / np.asarray(model["scale"])
    return float(1.0 / (1.0 + np.exp(-(model["intercept"] + float(np.dot(model["weights"], z))))))


def _house_tone(latest: str, previous: str | None, house: House | None):
    house = house or House()
    if not house.available:
        return None, None, "no model endpoint"
    try:
        n1, n2 = tone(house, latest, previous)
    except Exception as e:
        return None, None, f"House error {type(e).__name__} after {house.used} requests"
    return n1, n2, f"{house.used} House requests"


def text_overlay(samples: np.ndarray, assets: list[str], horizons: list[int], hist: dict, text_dir, asof: str,
                 house: House | None = None) -> tuple[np.ndarray, list[str]]:
    """samples: [n_draws, n_assets, n_horizons] (the engine's layout). Returns (samples, ledger)."""
    try:
        ust = [i for i, a in enumerate(assets) if str(a).startswith("UST_") and a in hist]
        if not ust:
            return samples, ["text layer: no UST targets, not applied"]
        statements = find_statements(pathlib.Path(text_dir), asof)
        if not statements:
            return samples, ["text layer: no FOMC statement in the corpus, not applied"]
        (d_latest, latest), previous = statements[-1], (statements[-2][1] if len(statements) > 1 else None)

        x1 = lexicon.tone(latest)
        x2 = x1 - lexicon.tone(previous) if previous else None
        nov = lexicon.novelty(latest, previous)
        ledger = [f"text layer: FOMC statement {d_latest}" + (f" (previous {statements[-2][0]})" if previous else ""),
                  f"  lexicon: hawkish-dovish tone {x1:+.2f}" + (f", change {x2:+.2f}" if x2 is not None else "")
                  + f"; sentiment (reported only): economic conditions {lexicon.econ_sentiment(latest):+.2f}, "
                  f"uncertainty {lexicon.uncertainty(latest):.1f} per 1k words"
                  + (f", wording change vs previous {nov:.2f}" if nov is not None else "")]

        n1, n2, note = _house_tone(latest, previous, house)
        if n1 is not None:
            model = _COEFS["house"]
            level, change = n1, (n2 if n2 is not None else model["mean"][2])
            ledger.append(f"  House tone: hawkishness {n1:.2f} (1-5), change vs previous "
                          f"{'n/a' if n2 is None else f'{n2:+.2f}'} ({note}) -> model 'house'")
        else:
            model = _COEFS["lexicon"]
            level, change = x1, (x2 if x2 is not None else model["mean"][2])
            ledger.append(f"  House tone unavailable ({note}) -> lexicon model")

        cap, k = _COEFS["mapping"]["cap_sd"], _COEFS["mapping"]["shift_sd_per_unit"]
        out = samples.copy()
        for ai in ust:
            a = assets[ai]
            p = _prob(model, [_m0_drift_sign(getattr(hist[a], "values", hist[a])), level, change])
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
