"""Event-selection robustness of the family width multiplier, on PUBLIC-HISTORY pseudo-cards only.
Guard: pseudo-cards whose (asset, as-of within +-7 days) coincide with any practice unit are dropped, so nothing is tied to
the practice units' hidden labels. Selection of 'shock' cards uses the realised |z| versus M0 (outcome-conditioned on purpose:
it mimics how event cards were authored); thresholds span mild to extreme because the true severity is unknown."""
import sys, glob, os, tomllib, json
sys.path.insert(0, "/Users/apple/Documents/agenthon/t2work")
import numpy as np, pandas as pd, exp
df = exp.init()
# --- guard: practice-unit (asset, as-of) pairs
pairs = []
for u in glob.glob("/Users/apple/Documents/agenthon/track2-forecasting-public/units/t2-*"):
    c = tomllib.load(open(u + "/card.toml", "rb")); asof = pd.Timestamp(c["provenance"]["data_cutoff"])
    for a in c["targets"]["asset_ids"]: pairs.append((a, asof))
def near(card):
    return any(a in card.assets and abs((card.asof - t).days) <= 7 for a, t in pairs)
keep = np.array([not near(c) for c in exp.CARDS]); single = np.array([len(c.cells) == 1 and c.panel != "macro" for c in exp.CARDS])
use = keep & single
print("single-cell pseudo-cards:", int(single.sum()), "after practice-overlap guard:", int(use.sum()))
ZS = exp.ZS
sets = {"all": np.ones(len(ZS), bool)}
for q in (0.67, 0.85, 0.95): sets[f"top{int(round((1-q)*100))}%"] = ZS >= np.quantile(ZS[use], q)
ks = [0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.7, 2.0]
tab = {}
for k in ks:
    P = {p: {"k": k, "k_family": {}} for p in ["rates", "fx", "factors"]}
    o = exp.evaluate(P, df, only=use)
    sc = pd.Series(o.score.values, index=o.index)
    tab[k] = {n: float(sc[m[sc.index]].mean()) for n, m in sets.items()}
T = pd.DataFrame(tab).T
print("\nmean score vs M0 (1.0 = M0; lower better) by width multiplier k and selection severity:\n", T.round(3).to_string())
reg = T - T.min()
print("\nregret vs best k within each selection level:\n", reg.round(3).to_string())
print("\nminimax-regret k over severities {top33,top15,top5}:", reg[["top33%", "top15%", "top5%"]].max(axis=1).idxmin(), "| over all four:", reg.max(axis=1).idxmin())
print("best k per level:", T.idxmin().to_dict())
