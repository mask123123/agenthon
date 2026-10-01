import sys, glob, tomllib
sys.path.insert(0, "/Users/apple/Documents/agenthon/t2work")
import numpy as np, pandas as pd, exp
df = exp.init()
pairs = []
for u in glob.glob("/Users/apple/Documents/agenthon/track2-forecasting-public/units/t2-*"):
    c = tomllib.load(open(u + "/card.toml", "rb")); asof = pd.Timestamp(c["provenance"]["data_cutoff"])
    for a in c["targets"]["asset_ids"]: pairs.append((a, asof))
keep = np.array([not any(a in c.assets and abs((c.asof - t).days) <= 7 for a, t in pairs) for c in exp.CARDS])
single = np.array([len(c.cells) == 1 and c.panel != "macro" for c in exp.CARDS]); use = keep & single
ZS = exp.ZS; sets = {"all": np.ones(len(ZS), bool)}
for q in (0.67, 0.85, 0.95): sets[f"top{int(round((1-q)*100))}%"] = ZS >= np.quantile(ZS[use], q)
variants = {
  "k1.0 nu8 (now)": dict(k=1.0, nu=8.0), "k1.0 nu4": dict(k=1.0, nu=4.0), "k1.0 nu3": dict(k=1.0, nu=3.0),
  "k1.35 nu8": dict(k=1.35, nu=8.0), "k1.35 nu4": dict(k=1.35, nu=4.0), "k1.5 nu4": dict(k=1.5, nu=4.0),
  "mix p.2 m2": dict(k=1.0, mix_p=0.2, mix_m=2.0), "mix p.3 m2": dict(k=1.0, mix_p=0.3, mix_m=2.0), "mix p.2 m3": dict(k=1.0, mix_p=0.2, mix_m=3.0),
  "mix p.3 m2.5": dict(k=1.0, mix_p=0.3, mix_m=2.5), "mix p.15 m3 k1.2": dict(k=1.2, mix_p=0.15, mix_m=3.0),
}
tab = {}
for n, kw in variants.items():
    o = exp.evaluate({p: {**kw, "k_family": {}} for p in ["rates", "fx", "factors"]}, df, only=use)
    sc = pd.Series(o.score.values, index=o.index); tab[n] = {s: float(sc[m[sc.index]].mean()) for s, m in sets.items()}
T = pd.DataFrame(tab).T; T["worst"] = T[["all", "top33%", "top15%", "top5%"]].max(axis=1); T["avg_shock"] = T[["top33%", "top15%", "top5%"]].mean(axis=1)
print(T.round(3).to_string())
