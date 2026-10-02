import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import review.eval_teammate as E
from backtest.evaluate import load_all, summarize, cluster_ci
from t2agent import m0

real, pseudo = load_all()
# 1) does the teammate's single-cell z equal M0's Z?
c = next(c for c in real if len(c["y"]) == 1)
import zlib
rng = np.random.default_rng(zlib.crc32(c["task"].unit_id.encode()) & 0x7FFFFFFF); z = rng.standard_normal((1000, 1))
rng2 = np.random.default_rng(zlib.crc32(c["task"].unit_id.encode()) & 0x7FFFFFFF); Z = rng2.standard_normal((500, 1))
print("single-cell z[:500] == M0 Z:", np.allclose(z[:500], Z))

orig = E.draws
def sorted_draws(task, override=None):
    t = task
    saved = t.card
    t.card = {"targets": {"asset_ids": sorted(t.assets)}}
    t.horizons = sorted(t.horizons)
    try: return orig(t, override)
    finally: t.card = saved

variants = {
    "as_submitted": (orig, None),
    "sorted_cells(CRN on multi)": (sorted_draws, None),
    "single_drift1.0": (orig, {"drift": 1.0}),
    "beta0": (orig, {"beta": 0.0}),
    "no_family_k": (orig, {"k_family": {}}),
    "gauss(nu=1e6)": (orig, {"nu": 1e6}),
}
for name, (fn, ov) in variants.items():
    E.draws = fn
    for lab, cs in (("real", real), ("pseudo", pseudo)):
        df = E.run(cs, ov); s = summarize(df, lab); lo, hi = cluster_ci(df)
        print(f"{name:28s} {lab:6s} mean={s['mean']:.3f} CI[{lo:.3f},{hi:.3f}] clip={s['n_clip']} "
              f"single={df[df.ncell==1].score.mean():.3f} multi={df[df.ncell>1].score.mean():.3f} "
              f"F1={s['F1']:.3f} F2={s['F2']:.3f} F3={s['F3']:.3f} F4={s['F4']:.3f}")
