"""Build prompts to test the House-model extraction fallback with a stand-in model. Positives: BLS documents with regex truth.
Negatives: wrong month / series not in the document (must answer found=false). Synthetic: foreign-format release excerpts
written by us (clearly synthetic) to test phrasing generalisation."""
import sys, glob, json, os, hashlib, random, re
sys.path.insert(0, "/Users/apple/Documents/agenthon/submission")
import numpy as np, pandas as pd
from t2agent import releases as R, llm_extract as LX
random.seed(11)
U = "/Users/apple/Documents/agenthon/track2-forecasting-public/units"
best = None
for f in glob.glob(U + "/*/macro_monthly.parquet"):
    d = pd.read_parquet(f)
    if best is None or d["date"].max() > best["date"].max(): best = d
P = best.assign(m=pd.to_datetime(best["date"]).dt.to_period("M").astype(str)).pivot(index="m", columns="asset", values="value")
seen, items, truth = set(), [], {}
def prompt(asset, last_month, last_val, want_month_text, doc_ts, text):
    return LX.USER.format(asset=asset, last_month=last_month, last_val=last_val, want_month=want_month_text,
                          text=f"[release dated {doc_ts}]\n" + re.sub(r"\s+", " ", text).strip()[:LX.MAX_CHARS_PER_DOC])
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
        if not r: continue
        prev = str(pd.Period(r["ref"], "M") - 1); nxt = str(pd.Period(r["ref"], "M") + 1)
        for asset, key, kind in [("CPI_ALL", "CPI_ALL_PCT", "pct_change"), ("CPI_CORE", "CPI_CORE_PCT", "pct_change"), ("UNRATE", "UNRATE", "level")]:
            if key in r["values"] and asset in P.columns and prev in P.index and np.isfinite(P.loc[prev, asset]):
                iid = f"P{len(items):03d}"
                items.append({"id": iid, "kind": "positive", "prompt": prompt(asset, prev, float(P.loc[prev, asset]), r["ref"], d["timestamp"], txt)})
                truth[iid] = {"type": "positive", "asset": asset, "kind": kind, "value": r["values"][key], "last_month": prev, "last_val": float(P.loc[prev, asset]), "sent": items[-1]["prompt"].split("TEXT:\n", 1)[1]}
                # negative control A: ask for the month AFTER the release month (text does not contain it)
                if random.random() < 0.35 and nxt in P.index:
                    jid = f"N{len(items):03d}"; ask_prev = r["ref"]
                    items.append({"id": jid, "kind": "neg_month", "prompt": prompt(asset, ask_prev, float(P.loc[ask_prev, asset]) if ask_prev in P.index and np.isfinite(P.loc[ask_prev, asset]) else float(P.loc[prev, asset]), nxt, d["timestamp"], txt)})
                    truth[jid] = {"type": "neg_month", "asset": asset, "last_month": ask_prev, "last_val": items[-1]["prompt"].split("is ")[1].split(".")[0], "sent": items[-1]["prompt"].split("TEXT:\n", 1)[1]}
        # negative control B: a series the document does not cover
        if random.random() < 0.3:
            asset = "PCE_ALL" if d["file"].startswith("cpi") else "CPI_ALL"
            if prev in P.index and asset in P.columns and np.isfinite(P.loc[prev, asset]):
                jid = f"M{len(items):03d}"
                items.append({"id": jid, "kind": "neg_series", "prompt": prompt(asset if asset == "PCE_ALL" else "RETAIL_SALES_US", prev, float(P.loc[prev, asset]), r["ref"], d["timestamp"], txt)})
                truth[jid] = {"type": "neg_series", "asset": asset, "last_month": prev, "last_val": float(P.loc[prev, asset]), "sent": items[-1]["prompt"].split("TEXT:\n", 1)[1]}
# synthetic foreign-format excerpts
SYN = [
 ("CPI_UK", "2024-06", 131.9, "2024-07", "ONS", "Consumer price inflation, UK: July 2024. The Consumer Prices Index (CPI) rose by 2.2% in the 12 months to July 2024, unchanged from June. On a monthly basis, CPI rose by 0.2% in July 2024, compared with a rise of 0.1% in July 2023. Services inflation was 5.2%.", "pct_change", 0.2),
 ("CPI_CA", "2024-06", 160.6, "2024-07", "StatCan", "The Consumer Price Index (CPI) rose 2.5% on a year-over-year basis in July, following a 2.7% increase in June. On a monthly basis, the CPI rose 0.4% in July, the same pace as in June. Shelter costs remained the largest contributor.", "pct_change", 0.4),
 ("HICP_EA", "2024-06", 124.2, "2024-07", "Eurostat", "Euro area annual inflation was 2.6% in July 2024, up from 2.5% in June. The euro area monthly inflation rate was 0.0% in July 2024, compared with 0.2% in June 2024. The highest contribution came from services.", "pct_change", 0.0),
 ("UNRATE_JP", "2024-06", 2.5, "2024-07", "MIC Japan", "Labour Force Survey, July 2024: the unemployment rate was 2.7 percent in July 2024, up 0.2 point from the previous month. The number of unemployed persons rose to 1.88 million.", "level", 2.7),
 ("CPI_JP", "2024-06", 108.2, "2024-07", "MIC Japan", "Consumer Price Index, Japan, July 2024: the core CPI (all items less fresh food) rose 2.7% year on year. No month-on-month figure is published in this summary.", None, None),
 ("UNRATE_UK", "2024-06", 4.4, "2024-07", "ONS", "Labour market overview, UK: July 2024. Employment rates were broadly flat. Vacancies fell by 8,000 in the three months to July 2024. Average weekly earnings grew by 4.5%.", None, None),
]
for asset, lm, lv, want, src, text, kind, val in SYN:
    iid = f"S{len(items):03d}"
    items.append({"id": iid, "kind": "synthetic", "prompt": prompt(asset, lm, lv, want, "2024-08-15", text)})
    truth[iid] = {"type": "synthetic", "asset": asset, "kind": kind, "value": val, "last_month": lm, "last_val": lv, "sent": items[-1]["prompt"].split("TEXT:\n", 1)[1]}
random.shuffle(items)
per = (len(items) + 5) // 6
for b in range(6):
    json.dump([{"id": i["id"], "system": LX.SYS, "user": i["prompt"]} for i in items[b * per:(b + 1) * per]], open(f"llm_ex/batch_{b}.json", "w"), ensure_ascii=False, indent=1)
json.dump(truth, open("llm_ex/truth_PRIVATE.json", "w"))
from collections import Counter
print("items:", len(items), Counter(i["kind"] for i in items), "| per batch ~", per)
