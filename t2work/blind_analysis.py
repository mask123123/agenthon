"""Analyse the blind LLM labels. Outcome per card: mean |z| (realised move / OUR engine's predicted sd) over a fixed set of
reference assets EXCLUDING the card's own target(s) -> a market-wide 'was the engine too narrow after this date' measure.
Pre-registered hypotheses: stress, pending_event, novelty, vol_next each positively related to abs-z. Controls: engine vol regime.
Significance: within-calendar-year permutation + Holm. Contamination split by guess_year accuracy / 'recognized'."""
import sys, glob, json, tomllib, os
sys.path.insert(0, "/Users/apple/Documents/agenthon/t2work"); sys.path.insert(0, "/Users/apple/Documents/agenthon/submission")
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from data import load_wide
from t2agent import engine

key = json.load(open("blind_key_PRIVATE.json"))
labs = []
for f in sorted(glob.glob("blind/out_*.json")):
    labs += json.load(open(f))
L = pd.DataFrame(labs); L = L[L.id.isin(key)].copy()
L["unit"] = L.id.map(lambda i: key[i]["unit"]); L["asof"] = pd.to_datetime(L.id.map(lambda i: key[i]["asof"]))
L["fam"] = L.id.map(lambda i: key[i]["family"]); L["year"] = L["asof"].dt.year
print("labelled items:", len(L), "| expected 104")

U = "/Users/apple/Documents/agenthon/track2-forecasting-public/units"
own = {}
for u in L.unit:
    c = tomllib.load(open(f"{U}/{u}/card.toml", "rb")); own[u] = set(c["targets"]["asset_ids"])
W = {k: load_wide(k) for k in ["rates", "fx", "factors"]}
REF = {"UST_2Y": ("rates", "level"), "UST_10Y": ("rates", "level"), "EUR": ("fx", "level"), "JPY": ("fx", "level"), "GBP": ("fx", "level"), "MKT": ("factors", "log_return")}
rows = []
for _, r in L.iterrows():
    for h in (21, 63):
        zs, vr = [], []
        for a, (pn, tt) in REF.items():
            if a in own[r.unit]: continue
            w = W[pn]; idx = w.index; i = idx.searchsorted(r["asof"], side="right") - 1
            if i < 400 or i + h >= len(idx): continue
            hist = w[a].iloc[:i + 1].dropna()
            y = float(w[a].iloc[i + h]) if tt == "level" else float(np.log1p(w[a].iloc[i + 1:i + h + 1].astype(float)).sum())   # LEVEL target: compare LEVELS (bug fix)
            try:
                res = engine.build_draws({a: hist}, [a], [h], tt, f"b-{r.unit}-{a}-{h}", 1000); x = res.samples.reshape(-1)
                st = (hist.diff() if tt == "level" else np.log1p(hist.astype(float))).dropna()
                zs.append(abs(y - x.mean()) / x.std()); vr.append(np.log(x.std() / (st.std() * np.sqrt(h))))
            except Exception: pass
        if len(zs) >= 2: rows.append(dict(unit=r.unit, h=h, absz=np.mean(zs), volreg=np.mean(vr)))
O = pd.DataFrame(rows); D = L.merge(O, on="unit")
print("rows:", len(D), "| units with outcomes:", D.unit.nunique())

def resid(y, X):
    X = np.column_stack([np.ones(len(y)), X]); b = np.linalg.lstsq(X, y, rcond=None)[0]; return y - X @ b
def pcorr(f, y, c): rf, ry, rc = rankdata(f), rankdata(y), rankdata(c); return float(np.corrcoef(resid(rf, rc), resid(ry, rc))[0, 1])
rng = np.random.default_rng(0)
def perm(f, y, c, yrs, B=2000):
    obs = pcorr(f, y, c); n = 0
    for _ in range(B):
        fp = f.copy()
        for yr in np.unique(yrs):
            m = yrs == yr; fp[m] = rng.permutation(f[m])
        n += abs(pcorr(fp, y, c)) >= abs(obs)
    return obs, (n + 1) / (B + 1)

# contamination
D["year_err"] = (D["guess_year"] - D["year"]).abs()
print("\nCONTAMINATION PROBE: median |guess_year - true year| = %.1f ; share within 1 year = %.0f%% ; share 'recognized' = %.0f%%" % (D.year_err.median(), 100 * (D.year_err <= 1).mean(), 100 * D.recognized.mean()))
res = []
for subset, mask in [("ALL", np.ones(len(D), bool)), ("NOT recognised", (D.recognized == 0).values), ("year guess off by >=2", (D.year_err >= 2).values)]:
    for h in (21, 63):
        d = D[mask & (D.h == h).values]
        if len(d) < 25: continue
        for feat in ["stress", "pending_event", "novelty", "vol_next"]:
            f = d[feat].values.astype(float)
            if np.std(f) == 0: continue
            obs, p = perm(f, d.absz.values, d.volreg.values, d.year.values)
            res.append(dict(subset=subset, h=h, feat=feat, n=len(d), raw=float(spearmanr(f, d.absz)[0]), partial=obs, p=p))
R = pd.DataFrame(res)
a = R[R.subset == "ALL"].sort_values("p").copy(); m = len(a); a["holm"] = np.maximum.accumulate([min(1, (m - i) * p) for i, p in enumerate(a.p)])
print("\nALL items (pre-registered 8 tests), Holm-adjusted:\n", a.round(3).to_string(index=False))
print("\nCleaner subsets (descriptive):\n", R[R.subset != "ALL"].sort_values(["subset", "p"]).round(3).to_string(index=False))
D.to_csv("blind_analysis_data_PRIVATE.csv", index=False)
