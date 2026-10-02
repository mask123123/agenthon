"""Offline scoring harness: normalized composite vs the exact M0 replica, on
  * real:   practice units whose outcomes resolve from stitched public panels
  * pseudo: practice-unit templates (assets, horizons, target type, panels) re-dated to random as-ofs

Score per card = sum_k w_k * (ours_k / M0_k), clipped at 4.0; single-cell cards use (5/7, 0, 2/7).
"""

from __future__ import annotations

import json
import pathlib
import pickle
import sys
import tomllib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from t2agent import engine, m0  # noqa: E402
from t2agent.task import Task, load_task  # noqa: E402

from qfbench2_common.scoring import crps  # noqa: E402
from qfbench2_track_forecasting.tail import tail_pinball  # noqa: E402

UNITS = ROOT.parents[1] / "track2-forecasting-public" / "units"
DATA = ROOT / "backtest" / "data"
TAIL = (0.01, 0.05, 0.95, 0.99)
CLIP = 4.0


def components(samples: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    return (
        crps.crps_marginal(samples, y),
        crps.variogram_score(samples, y, p=0.5) if len(y) > 1 else 0.0,
        tail_pinball(samples, y, TAIL),
    )


def normalized(ours, ref, ncell: int) -> tuple[float, float, float, float]:
    w = (0.5, 0.3, 0.2) if ncell > 1 else (0.5 / 0.7, 0.0, 0.2 / 0.7)
    r = [o / s if s > 0 else o for o, s in zip(ours, ref)]
    if ncell == 1:
        r[1] = 0.0
    return min(float(sum(wi * ri for wi, ri in zip(w, r))), CLIP), *r


# ----------------------------------------------------------------------------- datasets
def _history() -> dict[str, pd.DataFrame]:
    h = pd.read_parquet(DATA / "history.parquet")
    return {p: g[["date", "asset", "value"]].reset_index(drop=True) for p, g in h.groupby("panel")}


def _realize(series: pd.Series, asof: str, h: int, ttype: str) -> float:
    td = pd.Timestamp(str(np.busday_offset(asof, h, roll="backward"))[:10])
    if series.index.max() < td:
        return np.nan
    if ttype == "log_return":
        r = series[(series.index > pd.Timestamp(asof)) & (series.index <= td)]
        return float(np.log1p(r).sum()) if len(r) else np.nan
    prev = series[series.index <= td]
    return float(prev.iloc[-1]) if len(prev) and (td - prev.index[-1]).days <= 5 else np.nan


def real_cards() -> list[dict]:
    real = json.loads((DATA / "realized.json").read_text())
    out = []
    for uid, r in real.items():
        if not r["complete"]:
            continue
        t = load_task(UNITS / uid, UNITS / uid / "text", r["asof"])
        t.text_dir = None
        y = {(a, h): v for a, h, v, _ in r["cells"]}
        out.append({"task": t, "y": np.array([y[c] for c in t.cells()]), "kind": "real", "template": uid})
    return out


def pseudo_cards(k_per_template: int = 12, seed: int = 7, only=None, tag: str = "") -> list[dict]:
    """`only(task) -> bool` restricts the templates; `tag` names the cache when `only` is used."""
    cache = DATA / f"pseudo_k{k_per_template}_s{seed}{'_' + tag if tag else ''}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    hist = _history()
    series = {(p, a): pd.Series(g["value"].to_numpy(float), index=pd.to_datetime(g["date"]))
              for p, df in hist.items() for a, g in df.groupby("asset")}
    rng = np.random.default_rng(seed)
    out = []
    for u in sorted(UNITS.iterdir()):
        if not (u / "forecast_spec.json").is_file():
            continue
        stems = sorted(p.stem for p in list(u.glob("*.parquet")) + list((u / "panels").glob("*.parquet")))
        if any(s in ("em_transfer_early", "macro_monthly") for s in stems):
            continue
        card = tomllib.loads((u / "card.toml").read_text())
        tpl = load_task(u, None, card["provenance"]["data_cutoff"])
        if only is not None and not only(tpl):
            continue
        src ={a: next(s for s in sorted(tpl.panels) if (tpl.panels[s]["asset"] == a).any()) for a in tpl.assets}
        end = min(series[(src[a], a)].index.max() for a in tpl.assets)
        last_asof = pd.Timestamp(str(np.busday_offset(end.date(), -max(tpl.horizons) - 5, roll="backward")))
        days = pd.bdate_range("2002-01-02", last_asof)
        for d in rng.choice(len(days), size=k_per_template, replace=False):
            asof = str(days[d].date())
            panels = {s: hist[s][hist[s]["date"] <= asof] for s in stems}
            t = Task(unit_id=f"{tpl.unit_id}@{asof}", asof=asof, assets=tpl.assets, horizons=tpl.horizons,
                     target_type=tpl.target_type, target_frequency=tpl.target_frequency,
                     category=tpl.category, n_draws_min=tpl.n_draws_min, panels=panels)
            y = np.array([_realize(series[(src[a], a)], asof, h, t.target_type) for a, h in t.cells()])
            if np.isfinite(y).all():
                out.append({"task": t, "y": y, "kind": "pseudo", "template": tpl.unit_id})
    cache.write_bytes(pickle.dumps(out))
    return out


# ----------------------------------------------------------------------------- scoring
def _m0_ref(card: dict) -> tuple[float, float, float]:
    s, _ = m0.m0_draws(card["task"])
    return components(s, card["y"])


def _score_one(args):
    card, cfg = args
    t, y = card["task"], card["y"]
    try:
        s, cells = engine.forecast(t, cfg)
        ours = components(s, y)
        err = ""
    except Exception as e:  # a failed card is a 4.0 card, exactly as on the board
        ours, err = None, repr(e)[:120]
    ref = card["ref"]
    if ours is None:
        sc, rm, rj, rt = CLIP, np.nan, np.nan, np.nan
    else:
        sc, rm, rj, rt = normalized(ours, ref, len(y))
    return {"id": t.unit_id, "template": card["template"], "kind": card["kind"], "family": t.category[-2:],
            "ttype": t.target_type, "ncell": len(y), "year": int(t.asof[:4]), "asof": t.asof,
            "score": sc, "r_marg": rm, "r_joint": rj, "r_tail": rt, "err": err}


def attach_refs(cards: list[dict], name: str) -> list[dict]:
    cache = DATA / f"m0ref_{name}.pkl"
    refs = pickle.loads(cache.read_bytes()) if cache.exists() else {}
    todo = [c for c in cards if c["task"].unit_id not in refs]
    if todo:
        with ProcessPoolExecutor() as ex:
            for c, r in zip(todo, ex.map(_m0_ref, todo, chunksize=8)):
                refs[c["task"].unit_id] = r
        cache.write_bytes(pickle.dumps(refs))
    for c in cards:
        c["ref"] = refs[c["task"].unit_id]
    return cards


def run(cards: list[dict], cfg: engine.Config) -> pd.DataFrame:
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(_score_one, [(c, cfg) for c in cards], chunksize=8))
    return pd.DataFrame(rows)


def cluster_ci(df: pd.DataFrame, key: str = "asof", B: int = 1000, seed: int = 0) -> tuple[float, float]:
    g = df.groupby(key)["score"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(B, len(g)))
    s, n = g["sum"].to_numpy()[idx].sum(1), g["count"].to_numpy()[idx].sum(1)
    return tuple(np.percentile(s / n, [2.5, 97.5]))


def summarize(df: pd.DataFrame, label: str = "") -> dict:
    out = {"label": label, "n": len(df), "mean": df["score"].mean(), "median": df["score"].median(),
           "n_clip": int((df["score"] >= CLIP).sum()), "n_err": int((df["err"] != "").sum())}
    for f in ("F1", "F2", "F3", "F4"):
        out[f] = df.loc[df["family"] == f, "score"].mean()
    out["early(<2016)"] = df.loc[df["year"] < 2016, "score"].mean()
    out["late(>=2016)"] = df.loc[df["year"] >= 2016, "score"].mean()
    return out


def load_all(k: int = 12):
    real = attach_refs(real_cards(), "real")
    pseudo = attach_refs(pseudo_cards(k), f"pseudo_k{k}")
    return real, pseudo


if __name__ == "__main__":
    real, pseudo = load_all()
    print(f"real cards: {len(real)}   pseudo cards: {len(pseudo)}")
    for name, cfg in [("rw-gauss(EWMA20,no drift)", engine.Config())]:
        for lab, cs in (("real", real), ("pseudo", pseudo)):
            df = run(cs, cfg)
            print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summarize(df, lab).items()}))
