"""Step 1b (protocol frozen in RESEARCH_LOG.md): Nemotron tone vs lexicon, plus sentiment features, on rates cards."""

from __future__ import annotations

import json
import pathlib
import pickle
import re
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import backtest.step1_nlp as S  # noqa: E402
from backtest.evaluate import components, normalized  # noqa: E402

DATA = ROOT / "backtest" / "data"
ECON_POS = ["expand", "expansion", "strong", "strengthen", "solid", "robust", "improv", "gains", "rising", "pick up",
            "picked up", "firm", "healthy", "recover", "accelerat", "advanc", "brisk"]
ECON_NEG = ["weak", "declin", "deteriorat", "slow", "soft", "contraction", "contract", "recession", "decrease", "fell",
            "falling", "downturn", "strain", "stress", "turmoil", "losses", "sluggish", "subdued", "sagg", "faltering"]
UNCERT = ["uncertain", "risk", "unclear", "unpredictab", "volatil", "may ", "could ", "possib", "depend", "evolv",
          "monitor", "closely", "attentive"]


def words(t: str) -> list[str]:
    return re.findall(r"[a-z]+", t.lower())


def econ(t: str) -> float:
    s = t.lower()
    p, n = sum(s.count(w) for w in ECON_POS), sum(s.count(w) for w in ECON_NEG)
    return (p - n) / (p + n + 1)


def uncert(t: str) -> float:
    s = t.lower()
    return 1000.0 * sum(s.count(w) for w in UNCERT) / max(len(words(t)), 1)


def novelty(cur: str, prev: str | None) -> float:
    if not prev:
        return 0.0
    tri = lambda t: {tuple(w[i:i + 3]) for w in [words(t)] for i in range(len(w) - 2)}  # noqa: E731
    a, b = tri(cur), tri(prev)
    return 1.0 - len(a & b) / max(len(a | b), 1)


def statement_table() -> pd.DataFrame:
    st = json.loads((S.EXT / "fomc_statements.json").read_text())
    tone = json.loads((DATA / "nemotron_tone.json").read_text())
    dates = sorted(st)
    rows = []
    for i, d in enumerate(dates):
        prev = st[dates[i - 1]] if i else None
        t = tone.get(d, {})
        rows.append({"date": pd.Timestamp(d), "x1": S.tone(st[d]), "x3": econ(st[d]), "x4": uncert(st[d]),
                     "x5": novelty(st[d], prev), "n1": t.get("n1"), "n2": t.get("n2")})
    df = pd.DataFrame(rows)
    df["x2"] = df["x1"].diff().fillna(0.0)
    return df


def card_features(asof: str, tab: pd.DataFrame, tc: pd.DataFrame, st_list) -> dict:
    s1, s2, s3, _, _ = S.features(asof, tc, st_list)
    past = tab[tab["date"] <= pd.Timestamp(asof)]
    last = past.iloc[-1] if len(past) else None
    f = {"s1": s1, "s2": s2, "s3": s3}
    for c in ("x1", "x2", "x3", "x4", "x5", "n1", "n2"):
        v = None if last is None else last[c]
        f[c] = float(v) if v is not None and np.isfinite(v) else {"n1": 3.0}.get(c, 0.0)
    return f


def sanity(tab: pd.DataFrame, tc: pd.DataFrame) -> None:
    nxt = []
    for d in tab["date"]:
        w = tc[(tc["date"] > d) & (tc["date"] <= d + pd.Timedelta(days=90))]["change_bp"].sum()
        nxt.append(w)
    tab = tab.assign(next90=nxt)
    ok = tab.dropna(subset=["n1"])
    print(f"tone validity (corr with next-90-day Fed target change, n={len(ok)}): "
          f"lexicon x1 {ok['x1'].corr(ok['next90']):+.3f}   Nemotron n1 {ok['n1'].corr(ok['next90']):+.3f}   "
          f"corr(x1, n1) {ok['x1'].corr(ok['n1']):+.3f}   n2 vs next90 {ok['n2'].corr(ok['next90']):+.3f}")


if __name__ == "__main__":
    pseudo = pickle.loads((DATA / "step1_pseudo_rates.pkl").read_bytes())
    real = pickle.loads((DATA / "step1_real_rates.pkl").read_bytes())
    tc, st_list = S.load_policy()
    tab = statement_table()
    print(f"statements with Nemotron n1: {tab['n1'].notna().sum()}/{len(tab)}, n2: {tab['n2'].notna().sum()}")
    sanity(tab, tc)
    F = {r["id"]: card_features(r["asof"], tab, tc, st_list) for r in pseudo + real}

    def XY(R, cols):
        X, z = [], []
        for r in R:
            for i, v in enumerate(r["z"]):
                X.append([r["trend"][i]] + [F[r["id"]][c] for c in cols])
                z.append(v)
        return np.array(X), np.array(z)

    def deltas(R, predict, cols):
        sc = []
        for r in R:
            X = np.array([[r["trend"][i]] + [F[r["id"]][c] for c in cols] for i in range(len(r["z"]))])
            p = predict(X)
            shift = np.clip(0.25 * (2 * p - 1), -0.10, 0.10) * r["sd"]
            sc.append(normalized(components(r["base"] + shift[None, :], r["y"]), r["ref"], len(r["y"]))[0])
        s = pd.DataFrame({"m": [r["asof"][:7] for r in R], "score": sc}).groupby("m")["score"].agg(["sum", "count"])
        idx = np.random.default_rng(0).integers(0, len(s), (1000, len(s)))
        boot = s["sum"].to_numpy()[idx].sum(1) / s["count"].to_numpy()[idx].sum(1) - 1
        return float(np.mean(sc)) - 1, *np.percentile(boot, [2.5, 97.5])

    models = {"T": [], "T+L": ["x1", "x2"], "T+L+S": ["x1", "x2", "x3", "x4", "x5"], "T+N": ["n1", "n2"],
              "T+N+L+S": ["n1", "n2", "x1", "x2", "x3", "x4", "x5"]}
    cut = "2013-01-01"
    for nm, trf, tef in (("train<2013 test>=2013", lambda a: a < cut, lambda a: a >= cut),
                         ("train>=2013 test<2013", lambda a: a >= cut, lambda a: a < cut)):
        tr = [r for r in pseudo if trf(r["asof"])]
        te = [r for r in pseudo if tef(r["asof"])]
        te_r = [r for r in real if tef(r["asof"])]
        print(f"\n== {nm}: train {len(tr)} | test {len(te)} pseudo + {len(te_r)} real")
        for name, cols in models.items():
            X, z = XY(tr, cols)
            predict, w = S.fit_logit(X, z)
            Xt, zt = XY(te, cols)
            p = predict(Xt)
            d, lo, hi = deltas(te, predict, cols)
            dr = deltas(te_r, predict, cols)[0] if te_r else float("nan")
            print(f"  {name:8s} AUC {S.auc(p, zt):.3f} acc {((p > .5) == (zt == 1)).mean():.3f} | pseudo delta {d:+.4f} "
                  f"CI[{lo:+.4f},{hi:+.4f}] | real delta {dr:+.4f} | w={np.round(w[1:], 2)}")
