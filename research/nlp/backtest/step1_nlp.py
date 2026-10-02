"""Step 1 (protocol frozen in RESEARCH_LOG.md): does Fed policy state / statement tone predict the side of the
realized outcome relative to M0's centre on rates cards, well enough to pay under the M0-ratio metric?"""

from __future__ import annotations

import json
import pathlib
import pickle
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.evaluate import components, normalized, pseudo_cards, real_cards  # noqa: E402
from t2agent import m0  # noqa: E402

EXT = ROOT / "backtest" / "data" / "external" / "fed"
DATA = ROOT / "backtest" / "data"

HAWK = ["inflation pressures", "upside risk", "risks to inflation", "further firming", "additional firming",
        "policy firming", "ongoing increases", "further increases", "further gradual increases", "raise the target",
        "increase the target", "remain attentive to inflation", "inflation remains elevated", "elevated inflation",
        "highly attentive to inflation", "returning inflation to its 2 percent", "tight", "reduce its holdings",
        "reducing its holdings", "removal of policy accommodation", "measured pace", "strong", "robust",
        "heightened inflation", "inflation risks", "inflationary", "price pressures"]
DOVE = ["lower the target", "lowered the target", "reduce the target", "accommodative", "considerable period",
        "patient", "downside risks", "weaken", "deteriorat", "act as appropriate to sustain", "subdued",
        "below the committee's 2 percent", "extended period", "exceptionally low", "asset purchases", "reinvest",
        "slack", "strains", "turmoil", "disruption", "softening", "moderat", "uncertain", "decline", "contraction"]


def tone(text: str) -> float:
    t = text.lower()
    h = sum(t.count(p) for p in HAWK)
    d = sum(t.count(p) for p in DOVE)
    return (h - d) / (h + d + 1)


def load_policy():
    tc = pd.read_csv(EXT / "fed_target_changes.csv", parse_dates=["date"]).sort_values("date")
    st = json.loads((EXT / "fomc_statements.json").read_text())
    st = sorted((pd.Timestamp(k), tone(v)) for k, v in st.items())
    return tc, st


def features(asof: str, tc: pd.DataFrame, st: list) -> list[float]:
    t = pd.Timestamp(asof)
    past = tc[tc["date"] <= t]
    last = past.iloc[-1] if len(past) else None
    s1 = float(np.sign(last["change_bp"])) if last is not None else 0.0
    s2 = float(past[past["date"] > t - pd.Timedelta(days=180)]["change_bp"].sum()) / 100.0
    s3 = min((t - last["date"]).days / 365.0, 2.0) if last is not None else 2.0
    prior = [x for d, x in st if d <= t]
    x1 = prior[-1] if prior else 0.0
    x2 = (prior[-1] - prior[-2]) if len(prior) > 1 else 0.0
    return [s1, s2, s3, x1, x2]


def rows_for(card: dict) -> dict:
    t, y = card["task"], card["y"]
    base, cells, info = m0.m0_parts(t)
    ref = components(base, y)
    last = {a: float(t.history(a).iloc[-1]) for a in t.assets}
    trend = [float(np.sign(info["mean"][i] - last[a])) for i, (a, h) in enumerate(cells)]
    z = [1.0 if y[i] > info["mean"][i] else 0.0 for i in range(len(cells))]
    return {"id": t.unit_id, "asof": t.asof, "base": base, "sd": info["sd"], "y": y, "ref": ref, "z": z, "trend": trend}


def build(cards: list[dict], name: str) -> list[dict]:
    cache = DATA / f"step1_{name}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    with ProcessPoolExecutor() as ex:
        out = list(ex.map(rows_for, cards, chunksize=8))
    cache.write_bytes(pickle.dumps(out))
    return out


def fit_logit(X: np.ndarray, z: np.ndarray, C: float = 0.3, iters: int = 50):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Xs.shape[1])
    lam = np.r_[0.0, np.full(Xs.shape[1] - 1, 1.0 / C)]
    for _ in range(iters):  # Newton steps on the L2-penalized log-likelihood
        p = 1 / (1 + np.exp(-Xs @ w))
        g = Xs.T @ (p - z) + lam * w
        H = (Xs * (p * (1 - p))[:, None]).T @ Xs + np.diag(lam)
        w -= np.linalg.solve(H, g)
    return lambda Xn: 1 / (1 + np.exp(-np.c_[np.ones(len(Xn)), (Xn - mu) / sd] @ w)), w


def auc(p, z):
    order = np.argsort(p)
    r = np.empty(len(p))
    r[order] = np.arange(1, len(p) + 1)
    n1, n0 = z.sum(), len(z) - z.sum()
    return (r[z == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)


def evaluate(R, feats, cols, predict) -> tuple[float, float, float, float, float]:
    scores, P, Z = [], [], []
    for r in R:
        f = np.array([[feats[r["id"]][c] for c in cols]] * len(r["z"]))
        p = predict(f) if cols else np.full(len(r["z"]), 0.5)
        shift = np.clip(0.25 * (2 * p - 1), -0.10, 0.10) * r["sd"]
        scores.append(normalized(components(r["base"] + shift[None, :], r["y"]), r["ref"], len(r["y"]))[0])
        P += list(p)
        Z += r["z"]
    P, Z = np.array(P), np.array(Z)
    s = pd.DataFrame({"asof": [r["asof"][:7] for r in R], "score": scores})
    g = s.groupby("asof")["score"].agg(["sum", "count"])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(g), (1000, len(g)))
    boot = g["sum"].to_numpy()[idx].sum(1) / g["count"].to_numpy()[idx].sum(1)
    acc = float(((P > 0.5) == (Z == 1)).mean())
    return float(np.mean(scores)) - 1.0, *np.percentile(boot - 1.0, [2.5, 97.5]), acc, auc(P, Z)


if __name__ == "__main__":
    is_rates = lambda t: all(a.startswith("UST_") for a in t.assets)  # noqa: E731
    pseudo = build(pseudo_cards(40, 23, only=is_rates, tag="rates"), "pseudo_rates")
    real = build([c for c in real_cards() if is_rates(c["task"])], "real_rates")
    tc, st = load_policy()
    feats = {}
    for r in pseudo + real:
        f = features(r["asof"], tc, st)
        feats[r["id"]] = dict(zip(["s1", "s2", "s3", "x1", "x2"], f))
    print(f"pseudo rates cards: {len(pseudo)}   real rates cards: {len(real)}")
    models = {"1a state": ["s1", "s2", "s3"], "1b state+lexicon": ["s1", "s2", "s3", "x1", "x2"], "1c lexicon": ["x1", "x2"]}
    cut = "2013-01-01"
    for split_name, train_f, test_f in (("train<2013 test>=2013", lambda a: a < cut, lambda a: a >= cut),
                                        ("train>=2013 test<2013", lambda a: a >= cut, lambda a: a < cut)):
        tr = [r for r in pseudo if train_f(r["asof"])]
        te_p = [r for r in pseudo if test_f(r["asof"])]
        te_r = [r for r in real if test_f(r["asof"])]
        print(f"\n== {split_name}: train {len(tr)} pseudo | test {len(te_p)} pseudo + {len(te_r)} real")
        # control: trend sign as the 'signal' with P = 0.6 / 0.4
        for lab, te in (("pseudo", te_p), ("real", te_r)):
            sc, P, Z = [], [], []
            for r in te:
                p = 0.5 + 0.1 * np.array(r["trend"])
                shift = np.clip(0.25 * (2 * p - 1), -0.1, 0.1) * r["sd"]
                sc.append(normalized(components(r["base"] + shift[None, :], r["y"]), r["ref"], len(r["y"]))[0])
                P += list(p); Z += r["z"]
            print(f"  control trend-sign   {lab:6s} delta {np.mean(sc) - 1:+.4f}  acc {((np.array(P) > .5) == (np.array(Z) == 1)).mean():.3f}")
        for name, cols in models.items():
            X = np.array([[feats[r["id"]][c] for c in cols] for r in tr for _ in r["z"]])
            z = np.array([v for r in tr for v in r["z"]])
            predict, w = fit_logit(X, z)
            for lab, te in (("pseudo", te_p), ("real", te_r)):
                d, lo, hi, acc, a = evaluate(te, feats, cols, predict)
                print(f"  {name:18s} {lab:6s} delta {d:+.4f} CI[{lo:+.4f},{hi:+.4f}]  acc {acc:.3f}  AUC {a:.3f}  w={np.round(w[1:], 2)}")
