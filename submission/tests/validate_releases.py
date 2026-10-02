"""Validate the release parser on EVERY unique macro_release document of the practice corpora against the public monthly
panel (longest vintage). This checks the PARSER only: the reference month of a release is information that existed at its
publication date; no practice-card target is touched."""
import sys, glob, json, os, hashlib
sys.path.insert(0, "/Users/apple/Documents/agenthon/submission")
import numpy as np, pandas as pd
from t2agent import releases as R
U = "/Users/apple/Documents/agenthon/track2-forecasting-public/units"
best = None
for f in glob.glob(U + "/*/macro_monthly.parquet"):
    d = pd.read_parquet(f)
    if best is None or d["date"].max() > best["date"].max(): best = d
P = best.assign(m=pd.to_datetime(best["date"]).dt.to_period("M").astype(str)).pivot(index="m", columns="asset", values="value")
seen, rows, unparsed = set(), [], []
for u in sorted(glob.glob(U + "/t2-*")):
    p = u + "/text/corpus_index.json"
    if not os.path.exists(p): continue
    for d in json.load(open(p))["documents"]:
        if d["doc_type"] != "macro_release": continue
        f = u + "/text/" + d["file"]
        if not os.path.exists(f): continue
        txt = open(f, errors="replace").read(); h = hashlib.md5(txt.encode()).hexdigest()
        if h in seen: continue
        seen.add(h); r = R.parse_release(txt)
        if not r: unparsed.append((d["file"], d["timestamp"])); continue
        for a, v in r["values"].items():
            if a in ("CPI_ALL_PCT", "CPI_CORE_PCT"):
                col = a[:-4]; prev = str(pd.Period(r["ref"], "M") - 1)
                if col in P.columns and prev in P.index and r["ref"] in P.index and np.isfinite(P.loc[prev, col]) and np.isfinite(P.loc[r["ref"], col]):
                    rows.append(dict(file=d["file"], ref=r["ref"], asset=col, parsed=float(P.loc[prev, col]) * (1 + v / 100.0), panel=float(P.loc[r["ref"], col])))
                continue
            if a in P.columns and r["ref"] in P.index and np.isfinite(P.loc[r["ref"], a]):
                rows.append(dict(file=d["file"], ref=r["ref"], asset=a, parsed=v, panel=float(P.loc[r["ref"], a])))
            elif a == "NFP_CHANGE" and r["ref"] in P.index and R._month_after(r["ref"] + "-01") is not None:
                prev = str(pd.Period(r["ref"], "M") - 1)
                if prev in P.index and np.isfinite(P.loc[prev, "NFP"]) and np.isfinite(P.loc[r["ref"], "NFP"]):
                    rows.append(dict(file=d["file"], ref=r["ref"], asset="NFP", parsed=P.loc[prev, "NFP"] + r["values"].get("NFP_REVISION", 0) + v, panel=float(P.loc[r["ref"], "NFP"])))
print("unique macro_release docs:", len(seen), "| not parseable (title/format):", len(unparsed), unparsed[:6])
D = pd.DataFrame(rows); D["err"] = D.parsed - D.panel
tol = {"CPI_ALL": 0.45, "CPI_CORE": 0.45, "UNRATE": 0.15, "NFP": 160.0}
D["ok"] = [abs(e) <= tol[a] for e, a in zip(D.err, D.asset)]
print(D.groupby("asset").agg(n=("ok", "size"), exact_or_within_tol=("ok", "mean"), max_abs_err=("err", lambda s: np.abs(s).max())).round(4).to_string())
print("\nmismatches:\n", D[~D.ok][["file", "ref", "asset", "parsed", "panel", "err"]].round(3).to_string(index=False) if (~D.ok).any() else "none")
