"""Score the teammate's v2 engine (image sha256:2c9decfc...) in our harness, unchanged."""
import importlib.util, pathlib, sys, zlib
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.evaluate import load_all, components, normalized, summarize, cluster_ci
from t2agent import m0
spec = importlib.util.spec_from_file_location("tm_engine", ROOT / "review" / "teammate_v2_engine.py")
tm = importlib.util.module_from_spec(spec); sys.modules["tm_engine"] = tm; spec.loader.exec_module(tm)

def draws(task, override=None):
    assets, horizons = list(task.card.get("targets", {}).get("asset_ids", task.assets)) if task.card else task.assets, task.horizons
    hist = {a: task.history(a) for a in assets}
    ps = None
    if task.observation_periods:
        _, _, _, _, s = m0.estimate(task)
        ps = np.array([[s[(a, h)] for h in horizons] for a in assets])
    n = 2000 if task.category == "T2-F4" else 1000
    r = tm.build_draws(hist, assets, horizons, task.target_type, task.unit_id, n, family=task.category, panel_steps=ps, params=override)
    idx = {(a, h): (i, j) for i, a in enumerate(assets) for j, h in enumerate(horizons)}
    return np.stack([r.samples[:, idx[c][0], idx[c][1]] for c in task.cells()], axis=1)

def run(cards, override=None):
    rows = []
    for c in cards:
        t, y = c["task"], c["y"]
        try:
            sc, rm, rj, rt = normalized(components(draws(t, override), y), c["ref"], len(y)); err = ""
        except Exception as e:
            sc, rm, rj, rt, err = 4.0, np.nan, np.nan, np.nan, repr(e)[:80]
        rows.append({"id": t.unit_id, "family": t.category[-2:], "year": int(t.asof[:4]), "asof": t.asof, "ncell": len(y),
                     "score": sc, "r_marg": rm, "r_joint": rj, "r_tail": rt, "err": err})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    real, pseudo = load_all()
    for lab, cs in (("real", real), ("pseudo", pseudo)):
        df = run(cs)
        s = summarize(df, lab); lo, hi = cluster_ci(df)
        print("teammate_v2", lab, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}, f"CI[{lo:.3f},{hi:.3f}]",
              "single:", round(df[df.ncell == 1].score.mean(), 3), "multi:", round(df[df.ncell > 1].score.mean(), 3))
        df.to_csv(ROOT / "backtest" / "data" / f"teammate_v2_{lab}.csv", index=False)
