"""Stage 1.5. Pre-registered hypotheses (fixed BEFORE looking at results):
  H: higher density of uncertainty/alarm language in a document => realised |move| is larger relative to what OUR numeric
     engine predicted (z = (y - pred_mean)/pred_sd), i.e. the engine is too narrow after such documents.
Controls: the engine's current-vol regime (log pred_sd / long-run sd), via residualised rank-partial correlation;
significance by within-calendar-year permutation (kills year-level regime confounding).
Features come from text only; labels strictly after the document date; engine uses data <= document date."""
import sys, glob, json, os, re, hashlib
sys.path.insert(0, "/Users/apple/Documents/agenthon/t2work"); sys.path.insert(0, "/Users/apple/Documents/agenthon/submission")
import numpy as np, pandas as pd
from scipy.stats import rankdata
from data import load_wide
from t2agent import engine

U = "/Users/apple/Documents/agenthon/track2-forecasting-public/units"
UNC = ["uncertain", "volatil", "stress", "strain", "turmoil", "disruption", "concern", "risk", "pressure", "tension", "deteriorat", "adverse", "fragile"]
ALARM = ["crisis", "turmoil", "panic", "collapse", "recession", "default", "contagion", "severe", "sharp", "plunge", "tumble", "sell-off", "selloff", "emergency", "unprecedented"]

seen, docs = set(), []
for u in sorted(glob.glob(U + "/t2-*")):
    p = u + "/text/corpus_index.json"
    if not os.path.exists(p): continue
    for d in json.load(open(p))["documents"]:
        f = u + "/text/" + d["file"]
        if not os.path.exists(f): continue
        txt = open(f, errors="ignore").read(); h = hashlib.md5(txt.encode()).hexdigest()
        if h in seen: continue
        seen.add(h); docs.append(dict(date=pd.Timestamp(d["timestamp"]), type=d["doc_type"], text=txt))
def dens(t, words):
    t = re.sub(r"[^a-z\- ]", " ", t.lower()); n = max(len(t.split()), 1)
    return sum(t.count(w) for w in words) / n * 100
for d in docs: d["unc"] = dens(d["text"], UNC); d["alarm"] = dens(d["text"], ALARM)
print("unique docs:", len(docs))

W = {k: load_wide(k) for k in ["rates", "fx", "factors"]}
ASSETS = {"UST_2Y": ("rates", "level"), "UST_10Y": ("rates", "level"), "EUR": ("fx", "level"), "JPY": ("fx", "level"), "MKT": ("factors", "log_return")}
rows = []
for k, d in enumerate(docs):
    for a, (pn, tt) in ASSETS.items():
        w = W[pn]; idx = w.index; i = idx.searchsorted(d["date"], side="right") - 1
        if i < 400: continue
        hist = w[a].iloc[:i + 1].dropna()
        for h in (21, 63):
            if i + h >= len(idx): continue
            y = float(w[a].iloc[i + h]) if tt == "level" else float(np.log1p(w[a].iloc[i + 1:i + h + 1].astype(float)).sum())   # LEVEL target: compare LEVELS (bug fix)
            if not np.isfinite(y): continue
            try:
                r = engine.build_draws({a: hist}, [a], [h], tt, f"t-{k}-{a}-{h}", 1000)
                x = r.samples.reshape(-1)
                # long-run sd for the regime control
                st = (hist.diff() if tt == "level" else np.log1p(hist.astype(float))).dropna()
                lr_sd = float(st.std() * np.sqrt(h))
            except Exception:
                continue
            m, s = float(x.mean()), float(x.std())
            rows.append(dict(doc=k, date=d["date"], type=d["type"], asset=a, h=h, unc=d["unc"], alarm=d["alarm"], absz=abs(y - m) / s, volreg=np.log(s / lr_sd)))
R = pd.DataFrame(rows); R["year"] = R.date.dt.year
print("document-asset-horizon observations:", len(R))
# one row per document & horizon: average absz over assets (pooled outcome), control = average volreg
P = R.groupby(["doc", "type", "date", "year", "h"]).agg(unc=("unc", "first"), alarm=("alarm", "first"), absz=("absz", "mean"), volreg=("volreg", "mean"), n_assets=("asset", "size")).reset_index()

def resid(y, X):
    X = np.column_stack([np.ones(len(y)), X]); b = np.linalg.lstsq(X, y, rcond=None)[0]; return y - X @ b
def pcorr(f, y, c):
    rf, ry, rc = rankdata(f), rankdata(y), rankdata(c)
    return float(np.corrcoef(resid(rf, rc), resid(ry, rc))[0, 1])
rng = np.random.default_rng(0)
def perm_p(f, y, c, years, B=2000):
    obs = pcorr(f, y, c); cnt = 0
    for _ in range(B):
        fp = f.copy()
        for yr in np.unique(years):
            m = years == yr; fp[m] = rng.permutation(f[m])
        cnt += abs(pcorr(fp, y, c)) >= abs(obs)
    return obs, (cnt + 1) / (B + 1)

res = []
for typ in ["fomc_statement", "fomc_minutes", "beige_book", "cb_speech", "macro_release"]:
    for h in (21, 63):
        for feat in ("unc", "alarm"):
            d = P[(P.type == typ) & (P.h == h)].dropna()
            if len(d) < 25: continue
            f, y, c, yr = d[feat].values.astype(float), d.absz.values, d.volreg.values, d.year.values
            raw = float(np.corrcoef(rankdata(f), rankdata(y))[0, 1]); pc, p = perm_p(f, y, c, yr)
            tr, te = d.date < "2015-01-01", d.date >= "2015-01-01"
            r_tr = pcorr(f[tr], y[tr], c[tr]) if tr.sum() > 12 else np.nan; r_te = pcorr(f[te], y[te], c[te]) if te.sum() > 12 else np.nan
            res.append(dict(type=typ, h=h, feat=feat, n=len(d), raw_rho=raw, partial_rho=pc, p_within_year=p, train_pr=r_tr, test_pr=r_te))
T = pd.DataFrame(res).sort_values("p_within_year")
m = len(T); T["holm_p"] = [min(1.0, (m - i) * p) for i, p in enumerate(T.p_within_year)]; T["holm_p"] = np.maximum.accumulate(T.holm_p.values)
print("\n", T.round(3).to_string(index=False))
print("\nHolm-significant (<0.05):", int((T.holm_p < 0.05).sum()), "of", m, "| nominal p<0.05:", int((T.p_within_year < 0.05).sum()), "(chance would give ~%.1f)" % (0.05 * m))
T.to_csv("text_study2.csv", index=False); P.to_csv("text_study2_panel.csv", index=False)
