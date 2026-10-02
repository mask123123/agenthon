"""Which belief p to reweight: teammate engine draws, M0's own draws (common random numbers), or a 50/50 pool."""
import pathlib, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import review.eval_teammate as E
from backtest.evaluate import components, load_all, normalized
from backtest.ratio_aware import reweight
from t2agent import m0
GS = [0.0, 0.1, 0.25, 0.5]

def one(card):
    t, y = card["task"], card["y"]
    tm = E.draws(t); m0d, _ = m0.m0_draws(t)
    pool = np.vstack([tm[: len(tm) // 2], np.tile(m0d, (int(np.ceil(len(tm) / 2 / len(m0d))), 1))[: len(tm) - len(tm) // 2]])
    out = {"id": t.unit_id, "ncell": len(y)}
    for name, p in (("tm", tm), ("m0", np.tile(m0d, (2, 1))), ("pool", pool)):
        for g in GS:
            out[f"{name}_g{g}"] = normalized(components(reweight(p, m0d, g), y), card["ref"], len(y))[0]
    return out

if __name__ == "__main__":
    real, pseudo = load_all()
    for lab, cards in (("real", real), ("pseudo", pseudo)):
        with ProcessPoolExecutor() as ex:
            d = pd.DataFrame(list(ex.map(one, cards, chunksize=8)))
        line = []
        for name in ("tm", "m0", "pool"):
            for g in GS:
                c = f"{name}_g{g}"
                line.append(f"{c}: {d[c].mean():.4f} (s {d.loc[d.ncell==1, c].mean():.4f} / m {d.loc[d.ncell>1, c].mean():.4f})")
        print(f"== {lab}\n  " + "\n  ".join(line))
