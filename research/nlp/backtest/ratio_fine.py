"""Finer gamma grid, separately for single-cell and multi-cell cards (k = 1). Robust pick = min over max(real, pseudo)."""
import pathlib, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import review.eval_teammate as E
from backtest.evaluate import components, load_all, normalized, cluster_ci
from backtest.ratio_aware import reweight
from t2agent import m0
GS = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

def one(card):
    t, y = card["task"], card["y"]
    base = E.draws(t); m0d, _ = m0.m0_draws(t)
    out = {"id": t.unit_id, "asof": t.asof, "ncell": len(y)}
    for g in GS:
        out[f"g{g}"] = normalized(components(reweight(base, m0d, g), y), card["ref"], len(y))[0]
    return out

if __name__ == "__main__":
    real, pseudo = load_all()
    R = {}
    for lab, cards in (("real", real), ("pseudo", pseudo)):
        with ProcessPoolExecutor() as ex:
            R[lab] = pd.DataFrame(list(ex.map(one, cards, chunksize=8)))
        R[lab].to_pickle(ROOT / "backtest" / "data" / f"ratio_fine_{lab}.pkl")
    rows = []
    for gs in GS:
        for gm in GS:
            r = {"g_single": gs, "g_multi": gm}
            for lab in ("real", "pseudo"):
                d = R[lab]; s = np.where(d.ncell > 1, d[f"g{gm}"], d[f"g{gs}"])
                r[lab] = s.mean()
            r["worst"] = max(r["real"], r["pseudo"]); rows.append(r)
    t = pd.DataFrame(rows).sort_values("worst")
    print(t.head(8).to_string(index=False, float_format="{:.4f}".format))
    best = t.iloc[0]
    for lab in ("real", "pseudo"):
        d = R[lab].copy(); d["score"] = np.where(d.ncell > 1, d[f"g{best.g_multi}"], d[f"g{best.g_single}"])
        lo, hi = cluster_ci(d); d0 = d.assign(score=d["g0.0"]); diff = d.score - d["g0.0"]
        print(f"best ({best.g_single}/{best.g_multi}) {lab}: {d.score.mean():.4f} CI[{lo:.4f},{hi:.4f}] vs current {d0.score.mean():.4f}; "
              f"paired delta {diff.mean():+.4f}, better {int((diff<0).sum())} / worse {int((diff>0).sum())}")
