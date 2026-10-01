"""Executive summary: walk-forward backtest harness. Scores any forecaster(card, wide)->draws against M0
on the same pseudo-cards, reporting the mean normalized score (1.0 == M0, lower is better)."""
from __future__ import annotations
import numpy as np, pandas as pd
from core import Card, realized, m0_forecast, components, normalized, make_cards
from data import load_wide


def run(forecaster, cards, wides, verbose=True):
    rows = []
    for c in cards:
        w = wides[c.panel]
        y = realized(c, w)
        if y is None:
            continue
        single = len(c.cells) == 1
        ref = components(m0_forecast(c, w), y, single)
        if single:
            ref["joint"] = 1.0
        try:
            x = forecaster(c, w)
            comp = components(x, y, single)
            s = normalized(comp, ref, single)
        except Exception as e:  # a crash is 4.0 in the real scoring
            s = 4.0
        rows.append(dict(uid=c.uid, panel=c.panel, kind=c.family_hint, asof=c.asof, single=single, score=min(s, 4.0)))
    df = pd.DataFrame(rows)
    return df


def summarize(df, label=""):
    by = df.groupby(["panel", "kind"]).score.mean().round(3).to_dict()
    print(f"{label:28s} mean={df.score.mean():.4f}  n={len(df)}  ", by)


if __name__ == "__main__":
    wides = {k: load_wide(k) for k in ["rates", "fx", "factors"]}
    cards = make_cards(wides, step=63)
    print("cards:", len(cards))
    df = run(lambda c, w: m0_forecast(c, w, 500), cards, wides)
    summarize(df, "M0 vs M0 (should be 1.0)")
