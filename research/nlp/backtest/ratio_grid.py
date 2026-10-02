"""Grid over (k = widen engine draws about the median before reweighting, gamma = ratio-aware strength), by family."""
import pathlib, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import review.eval_teammate as E
from backtest.evaluate import components, load_all, normalized
from backtest.ratio_aware import reweight
from t2agent import m0
GRID = [(k, g) for k in (1.0, 1.15, 1.3, 1.5) for g in (0.0, 0.25, 0.5, 0.75)]

def one(card):
    t, y = card["task"], card["y"]
    base = E.draws(t); m0d, _ = m0.m0_draws(t); med = np.median(base, axis=0)
    out = {"id": t.unit_id, "fam": t.category[-2:], "ncell": len(y)}
    for k, g in GRID:
        x = reweight(med + k * (base - med), m0d, g)
        out[(k, g)] = normalized(components(x, y), card["ref"], len(y))[0]
    return out

if __name__ == "__main__":
    real, pseudo = load_all()
    res = {}
    for lab, cards in (("real", real), ("pseudo", pseudo)):
        with ProcessPoolExecutor() as ex:
            res[lab] = pd.DataFrame(list(ex.map(one, cards, chunksize=8)))
        res[lab].to_pickle(ROOT / "backtest" / "data" / f"ratio_grid_{lab}.pkl")
    rows = []
    for k, g in GRID:
        r, p = res["real"], res["pseudo"]
        row = {"k": k, "gamma": g, "real": r[(k, g)].mean(), "pseudo": p[(k, g)].mean()}
        for f in ("F1", "F2", "F3", "F4"):
            row[f"real_{f}"] = r.loc[r.fam == f, (k, g)].mean()
        row["worst"] = max(row["real"], row["pseudo"])
        rows.append(row)
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format):
        print(pd.DataFrame(rows).sort_values("worst").to_string(index=False))
