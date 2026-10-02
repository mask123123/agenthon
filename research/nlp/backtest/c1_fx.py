"""Step C1 (protocol frozen in RESEARCH_LOG.md): does Fed tone help forecast USD FX on top of the teammate engine?"""

from __future__ import annotations

import pathlib
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import backtest.step1_nlp as S  # noqa: E402
import backtest.step1b_compare as B  # noqa: E402
import review.eval_teammate as E  # noqa: E402
from backtest.evaluate import components, normalized, pseudo_cards, real_cards  # noqa: E402

DATA = ROOT / "backtest" / "data"
G10 = {"AUD", "CAD", "CHF", "DKK", "EUR", "GBP", "JPY", "NOK", "NZD", "SEK"}
ORIENT = {"AUD": -1, "EUR": -1, "GBP": -1, "NZD": -1, "CAD": 1, "CHF": 1, "DKK": 1, "JPY": 1, "NOK": 1, "SEK": 1}


def is_fx(t) -> bool:
    return all(a in G10 for a in t.assets)


def prepare():
    cache = DATA / "c1_fx.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    cards = pseudo_cards(40, 29, only=is_fx, tag="fx") + [c for c in real_cards() if is_fx(c["task"])]
    out = []
    for c in cards:
        r = S.rows_for(c)
        d = E.draws(c["task"])
        r.update(cells=c["task"].cells(), tm_draws=d, tm_sd=d.std(0), tm_comp=components(d, c["y"]))
        out.append(r)
    cache.write_bytes(pickle.dumps(out))
    return out


def X_of(r, F):
    f = F[r["id"]]
    return np.array([[r["trend"][i], ORIENT[a] * (f["n1"] - 3.0), ORIENT[a] * f["n2"]]
                     for i, (a, h) in enumerate(r["cells"])])


def delta(te, predict, F, base_key):
    base, new = [], []
    for r in te:
        p = predict(X_of(r, F))
        if base_key == "tm":
            draws, sd, comp = r["tm_draws"], r["tm_sd"], r["tm_comp"]
        else:
            draws, sd, comp = r["base"], r["sd"], r["ref"]
        shift = np.clip(0.25 * (2 * p - 1), -0.1, 0.1) * sd
        base.append(normalized(comp, r["ref"], len(r["y"]))[0])
        new.append(normalized(components(draws + shift[None, :], r["y"]), r["ref"], len(r["y"]))[0])
    d = np.array(new) - np.array(base)
    g = pd.DataFrame({"m": [r["asof"][:7] for r in te], "d": d}).groupby("m")["d"].agg(["sum", "count"])
    idx = np.random.default_rng(0).integers(0, len(g), (1000, len(g)))
    lo, hi = np.percentile(g["sum"].to_numpy()[idx].sum(1) / g["count"].to_numpy()[idx].sum(1), [2.5, 97.5])
    return float(np.mean(base)), float(d.mean()), float(lo), float(hi)


if __name__ == "__main__":
    R = prepare()
    tc, st = S.load_policy()
    tab = B.statement_table()
    F = {r["id"]: B.card_features(r["asof"], tab, tc, st) for r in R}
    pseudo = [r for r in R if "@" in r["id"]]
    real = [r for r in R if "@" not in r["id"]]
    print(f"FX pseudo cards {len(pseudo)}, real {len(real)}")
    cut = "2013-01-01"
    pooled_real = []
    for nm, trf, tef in (("train<2013", lambda a: a < cut, lambda a: a >= cut), ("train>=2013", lambda a: a >= cut, lambda a: a < cut)):
        tr = [r for r in pseudo if trf(r["asof"])]
        X = np.vstack([X_of(r, F) for r in tr])
        z = np.array([v for r in tr for v in r["z"]])
        predict, w = S.fit_logit(X, z)
        te_p = [r for r in pseudo if tef(r["asof"])]
        te_r = [r for r in real if tef(r["asof"])]
        Xt = np.vstack([X_of(r, F) for r in te_p])
        zt = np.array([v for r in te_p for v in r["z"]])
        print(f"\n== {nm}  w(trend, o*n1c, o*n2)={np.round(w[1:], 3)}  test AUC {S.auc(predict(Xt), zt):.3f}")
        for base_key in ("tm", "m0"):
            b, d, lo, hi = delta(te_p, predict, F, base_key)
            line = f"  base={base_key:2s} pseudo {b:.4f} delta {d:+.4f} CI[{lo:+.4f},{hi:+.4f}] n={len(te_p)}"
            if te_r:
                br, dr, _, _ = delta(te_r, predict, F, base_key)
                line += f" | real {br:.4f} delta {dr:+.4f} n={len(te_r)}"
                if base_key == "tm":
                    pooled_real += [dr] * len(te_r)
            print(line)
    print(f"\npooled real delta (teammate base): {np.mean(pooled_real):+.4f}")
