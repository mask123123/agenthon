"""Executive summary: pseudo-card generator, official-composite scoring, and the M0 baseline
(docs/M0-BASELINE.md) for our own walk-forward backtest. Scoring math is imported from the
organizers' toolkit (not re-implemented); only the glue is ours."""
from __future__ import annotations
import zlib
from dataclasses import dataclass
import numpy as np
import pandas as pd
from qfbench2_common.scoring import crps as C
from qfbench2_track_forecasting.tail import tail_pinball

LEVELS = (0.01, 0.05, 0.95, 0.99)


@dataclass
class Card:
    uid: str                 # stands in for card.toml [task].id (M0 seed)
    panel: str               # rates | fx | factors
    asof: pd.Timestamp
    assets: list[str]        # sorted
    horizons: list[int]      # sorted ascending, business days
    target_type: str         # level | log_return
    family_hint: str = ""

    @property
    def cells(self):
        return [(a, h) for a in self.assets for h in self.horizons]   # asset id, then horizon asc


def realized(card: Card, wide: pd.DataFrame) -> np.ndarray | None:
    """Realized value per cell: level at row t+h, or cumulative log(1+r) over h rows."""
    idx = wide.index
    t = idx.searchsorted(card.asof, side="right") - 1
    out = []
    for a, h in card.cells:
        if t + h >= len(idx):
            return None
        s = wide[a]
        if card.target_type == "level":
            v = s.iloc[t + h]
        else:
            v = np.log1p(s.iloc[t + 1: t + h + 1].astype(float)).sum()
        if not np.isfinite(v):
            return None
        out.append(float(v))
    return np.array(out)


def history(card: Card, wide: pd.DataFrame, n: int | None = None) -> pd.DataFrame:
    h = wide.loc[:card.asof, card.assets]
    return h.iloc[-n:] if n else h


def m0_forecast(card: Card, wide: pd.DataFrame, n_draws: int = 500) -> np.ndarray:
    """M0 as specified in docs/M0-BASELINE.md (daily panels). Returns [n_draws, n_cells]."""
    h = history(card, wide, 300)
    steps = h.diff() if card.target_type == "level" else h.copy()
    steps = steps.dropna()                               # date-intersection of every asset's steps
    mu = steps.mean().values
    sig = np.atleast_2d(np.cov(steps.values.T))
    ai = {a: i for i, a in enumerate(card.assets)}
    anchor = np.array([h[a].iloc[-1] if card.target_type == "level" else 0.0 for a in card.assets])
    cells = card.cells
    d = len(cells)
    mean = np.array([anchor[ai[a]] + s * mu[ai[a]] for a, s in cells])
    cov = np.empty((d, d))
    for i, (a, si) in enumerate(cells):
        for j, (b, sj) in enumerate(cells):
            cov[i, j] = min(si, sj) * sig[ai[a], ai[b]]
    cov = cov + 1e-10 * np.eye(d)
    try:
        L = np.linalg.cholesky(cov + 1e-9 * np.eye(d))
    except np.linalg.LinAlgError:
        L = np.diag(np.sqrt(np.diag(cov + 1e-9 * np.eye(d))))
    rng = np.random.default_rng(zlib.crc32(card.uid.encode()) & 0x7FFFFFFF)
    Z = rng.standard_normal((n_draws, d))
    return mean + Z @ L.T


def components(samples: np.ndarray, y: np.ndarray, single: bool) -> dict:
    marg = C.crps_marginal(samples, y)
    jnt = 0.0 if single else C.variogram_score(samples, y, p=0.5)
    tail = tail_pinball(samples, y, LEVELS)
    return {"marginal": marg, "joint": jnt, "tail": tail}


def normalized(comp: dict, ref: dict, single: bool) -> float:
    """Leaderboard-style score: components divided by M0's, effective weights by card shape."""
    w = (0.714286, 0.0, 0.285714) if single else (0.5, 0.3, 0.2)
    j = 0.0 if single else comp["joint"] / ref["joint"]
    return w[0] * comp["marginal"] / ref["marginal"] + w[1] * j + w[2] * comp["tail"] / ref["tail"]


def make_cards(wides: dict, start="2004-01-01", end_margin=0, step=21, seed=0) -> list[Card]:
    """Pseudo-cards on a regular as-of grid. Mix of single-cell and multi-cell shapes."""
    cards: list[Card] = []
    hs = [21, 63, 126]
    for pname, wide in wides.items():
        ttype = "log_return" if pname == "factors" else "level"
        idx = wide.dropna(how="all").index
        asofs = idx[(idx >= start)][::step]
        assets = sorted(wide.columns)
        for t in asofs:
            for a in assets:                               # single-cell cards
                for h in hs:
                    cards.append(Card(f"{pname}-{a}-{h}-{t.date()}", pname, t, [a], [h], ttype, "single"))
            if pname == "rates":                           # joint curve cards
                cards.append(Card(f"{pname}-curve21-{t.date()}", pname, t, ["UST_10Y", "UST_2Y", "UST_30Y", "UST_5Y"], [21], ttype, "joint"))
                cards.append(Card(f"{pname}-2y-multi-{t.date()}", pname, t, ["UST_2Y"], [63, 126], ttype, "multi-h"))
            if pname == "fx":
                cards.append(Card(f"{pname}-g4-{t.date()}", pname, t, ["EUR", "GBP", "JPY", "CHF"], [63], ttype, "joint"))
            if pname == "factors":
                cards.append(Card(f"{pname}-all-{t.date()}", pname, t, assets, [21], ttype, "joint"))
    return cards
