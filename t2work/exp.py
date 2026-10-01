"""Executive summary: parameter search harness. Uses the SHIPPED engine (t2agent.engine) so what is tested
is what runs. Caches realized outcomes + M0 reference components per card; evaluates param dicts in parallel.
Selection uses ONLY as-of < 2015; as-of >= 2015 is held out. 'shock' sets simulate event-card selection."""
from __future__ import annotations
import sys, itertools, json, os
sys.path.insert(0, "/Users/apple/Documents/agenthon/submission")
import numpy as np, pandas as pd
from multiprocessing import get_context
from data import load_wide
from core import Card, make_cards, realized, m0_forecast, components, normalized
from t2agent import engine

TREND = pd.Timestamp("2015-01-01")
WIDES = {k: load_wide(k) for k in ["rates", "fx", "factors", "macro"]}


def macro_cards(step=3):
    w = WIDES["macro"]; idx = w.index; out = []
    for t in idx[idx >= "2004-01-01"][::step]:
        for a in sorted(w.columns):
            for h in (2, 3, 8):
                out.append(Card(f"macro-{a}-{h}-{t.date()}", "macro", t, [a], [h], "level", "single"))
    return out


def extra_multi(step=63):
    """Shapes common in the real roster: several assets x two horizons (F3), one asset x two horizons (F1)."""
    out = []
    spec = {"rates": ([["UST_10Y", "UST_2Y"], ["UST_10Y", "UST_2Y", "UST_30Y"], ["UST_10Y"]], [[63, 126], [21, 63], [63, 189]], "level"),
            "fx": ([["EUR", "JPY", "GBP"], ["EUR"], ["JPY", "CHF"]], [[21, 63], [63, 126], [63, 189]], "level"),
            "factors": ([["MKT", "HML", "MOM"], ["MKT"], ["MOM", "BAB"]], [[21, 63], [63, 126], [21, 63]], "log_return")}
    for pn, (aset, hz, tt) in spec.items():
        w = WIDES[pn]; idx = w.dropna(how="all").index
        for t in idx[idx >= "2004-01-01"][::step]:
            for a, h in zip(aset, hz):
                out.append(Card(f"{pn}-mah-{'+'.join(a)}-{h[0]}-{h[1]}-{t.date()}", pn, t, sorted(a), h, tt, "mah"))
    return out


def all_cards():
    base = make_cards({k: WIDES[k] for k in ["rates", "fx", "factors"]}, start="2004-01-01", step=63)
    return base + extra_multi() + macro_cards()


def _forecast(card: Card, params: dict, n=1000):
    w = WIDES[card.panel]
    hist = {a: w[a].loc[:card.asof].dropna() for a in card.assets}
    ps = np.array([[h for h in card.horizons] for _ in card.assets], dtype=float) if card.panel == "macro" else None
    r = engine.build_draws(hist, card.assets, card.horizons, card.target_type, card.uid, n,
                           family="", panel_steps=ps, params=params)
    s = r.samples                                   # [n, A, H] -> cells in card.cells order (asset, horizon)
    return s.reshape(s.shape[0], -1)


# ---- cache ------------------------------------------------------------------------------------
CARDS = []; Y = []; REF = []; FLAGS = []
def _prep(c):
    w = WIDES[c.panel]; y = realized(c, w)
    if y is None: return None
    single = len(c.cells) == 1
    x = m0_forecast(c, w)
    ref = components(x, y, single)
    if single: ref["joint"] = 1.0
    z = float(np.abs((y - x.mean(0)) / x.std(0)).max())
    return c, y, ref, z, single


def _score_one(args):
    i, params = args
    c, y, ref, single = CARDS[i], Y[i], REF[i], len(CARDS[i].cells) == 1
    try:
        s = normalized(components(_forecast(c, params), y, single), ref, single)
    except Exception:
        s = 4.0
    return min(s, 4.0)


def init():
    global CARDS, Y, REF, ZS
    cards = all_cards()
    ctx = get_context("fork")
    with ctx.Pool(8) as p:
        prepped = [r for r in p.map(_prep, cards, chunksize=200) if r]
    CARDS = [r[0] for r in prepped]; Y = [r[1] for r in prepped]; REF = [r[2] for r in prepped]
    ZS = np.array([r[3] for r in prepped])
    thr = np.quantile(ZS, 0.67)
    df = pd.DataFrame({"panel": [c.panel for c in CARDS], "asof": [c.asof for c in CARDS], "shock": ZS >= thr})
    df["train"] = df["asof"] < TREND
    return df


def evaluate(params_by_panel: dict, df: pd.DataFrame, panels=None, only=None):
    """params_by_panel: {panel: params}. Returns df with 'score' per card."""
    idx = [i for i, c in enumerate(CARDS) if (panels is None or c.panel in panels) and (only is None or only[i])]
    ctx = get_context("fork")
    with ctx.Pool(8) as p:
        sc = p.map(_score_one, [(i, params_by_panel.get(CARDS[i].panel, {})) for i in idx], chunksize=100)
    out = df.iloc[idx].copy(); out["score"] = sc
    return out


def summarize(out):
    g = lambda m: float(out[m].score.mean()) if m.any() else float("nan")
    tr, sh = out.train, out.shock
    return {"train_all": g(tr), "train_shock": g(tr & sh), "test_all": g(~tr), "test_shock": g(~tr & sh),
            "obj": 0.5 * g(tr) + 0.5 * g(tr & sh)}
