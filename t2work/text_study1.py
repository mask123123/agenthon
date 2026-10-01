"""Stage 1 (clean, mechanical): do FOMC statements carry information the price panel does not?
Features: computed from text only (a-priori word lists, change vs previous statement, novelty).
Labels: strictly AFTER the statement date (close of day t onward). No LLM, no hindsight."""
import sys, glob, json, os, re, hashlib
sys.path.insert(0, "/Users/apple/Documents/agenthon/t2work")
import numpy as np, pandas as pd
from collections import Counter
from scipy.stats import spearmanr
from data import load_wide

U = "/Users/apple/Documents/agenthon/track2-forecasting-public/units"
seen, docs = set(), []
for u in sorted(glob.glob(U + "/t2-*")):
    p = u + "/text/corpus_index.json"
    if not os.path.exists(p): continue
    for d in json.load(open(p))["documents"]:
        if d["doc_type"] != "fomc_statement": continue
        f = u + "/text/" + d["file"]
        if not os.path.exists(f): continue
        txt = open(f, errors="ignore").read(); h = hashlib.md5(txt.encode()).hexdigest()
        if h in seen: continue
        seen.add(h); docs.append((pd.Timestamp(d["timestamp"]), txt))
docs.sort(key=lambda x: x[0]); print("unique FOMC statements:", len(docs))

HAWK = ["tighten", "firming", "firm ", "raise", "increase in the target", "inflation pressures", "upside risks to inflation", "elevated", "strong", "robust", "solid", "removal of", "reduce the pace", "rapid", "above", "persistent", "overheating"]
DOVE = ["accommodat", "patient", "weak", "downside risks", "ease", "easing", "lower", "reduce the target", "subdued", "slack", "soft", "decline", "below", "stimulus", "support"]
UNC = ["uncertain", "volatil", "stress", "strain", "turmoil", "disruption", "concern", "risk", "pressure", "tension", "deteriorat", "adverse", "fragile"]
def clean(t):  # strip boilerplate that is not policy content
    t = re.sub(r"(?s)Voting for the FOMC monetary policy action were.*", "", t)
    return re.sub(r"[^a-z ]", " ", t.lower())
def score(t, words): return sum(t.count(w) for w in words) / max(len(t.split()), 1) * 100
def vec(t): return Counter(t.split())
def cos(a, b):
    ks = set(a) | set(b); x = np.array([a.get(k, 0) for k in ks], float); y = np.array([b.get(k, 0) for k in ks], float)
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))

rows, prev = [], None
for t, raw in docs:
    c = clean(raw); v = vec(c)
    r = dict(date=t, hawk=score(c, HAWK), dove=score(c, DOVE), unc=score(c, UNC), words=len(c.split()))
    r["tone"] = r["hawk"] - r["dove"]
    r["novelty"] = 0.0 if prev is None else 1 - cos(v, prev[1])
    r["d_tone"] = np.nan if prev is None else r["tone"] - prev[2]
    r["d_unc"] = np.nan if prev is None else r["unc"] - prev[3]
    rows.append(r); prev = (t, v, r["tone"], r["unc"])
F = pd.DataFrame(rows)

R = load_wide("rates"); idx = R.index
def outcomes(t, h):
    i = idx.searchsorted(t, side="right") - 1        # last close on/before the statement date (includes its reaction)
    if i < 300 or i + h >= len(idx): return None
    out = {}
    for a in ["UST_2Y", "UST_10Y"]:
        s = R[a].diff().dropna()
        sd_trail = s.iloc[i - 300:i].std()           # trailing 300-step sd (what our numeric model would use)
        chg = R[a].iloc[i + h] - R[a].iloc[i]
        out[f"{a}_chg{h}"] = chg
        out[f"{a}_absz{h}"] = abs(chg) / (sd_trail * np.sqrt(h))      # realised move in units of the numeric model's own sd
    rv_fwd = R["UST_2Y"].diff().iloc[i + 1:i + 1 + h].std(); rv_back = R["UST_2Y"].diff().iloc[i - 60:i].std()
    out[f"volratio{h}"] = rv_fwd / rv_back
    return out
for h in (21, 63):
    O = [outcomes(t, h) for t in F.date]
    for k in (set().union(*[set(o) for o in O if o])):
        F[k] = [o.get(k, np.nan) if o else np.nan for o in O]
F = F.dropna(subset=["UST_2Y_chg21"]).reset_index(drop=True)
print("statements with outcomes:", len(F), F.date.min().date(), "->", F.date.max().date())

rng = np.random.default_rng(0)
def test(x, y, name):
    m = x.notna() & y.notna(); xv, yv = x[m].values, y[m].values
    if len(xv) < 20: return
    rho = spearmanr(xv, yv)[0]
    perm = [abs(spearmanr(xv, rng.permutation(yv))[0]) for _ in range(2000)]
    p = float(np.mean(np.array(perm) >= abs(rho)))
    print(f"  {name:38s} n={len(xv):3d} rho={rho:+.3f} perm-p={p:.3f}" + ("  <--" if p < 0.05 else ""))
print("\nA) DIRECTION: does the statement predict the subsequent yield change (after its own-day reaction)?")
for f in ["tone", "d_tone", "novelty"]:
    for y in ["UST_2Y_chg21", "UST_2Y_chg63", "UST_10Y_chg21", "UST_10Y_chg63"]: test(F[f], F[y], f"{f} -> {y}")
print("\nB) WIDTH: does uncertainty language / rewrite size predict |move| relative to what our numeric model expects?")
for f in ["unc", "d_unc", "novelty", "tone"]:
    for y in ["UST_2Y_absz21", "UST_2Y_absz63", "UST_10Y_absz21", "volratio21", "volratio63"]: test(F[f], F[y], f"{f} -> {y}")
F.to_csv("text_study1.csv", index=False)
