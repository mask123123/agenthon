"""Score the stand-in extraction answers: run the SAME verifier the image uses, then compare accepted values with regex truth."""
import sys, glob, json, re
sys.path.insert(0, "/Users/apple/Documents/agenthon/submission")
import numpy as np, pandas as pd
from collections import Counter, defaultdict
from t2agent import llm_extract as LX
truth = json.load(open("llm_ex/truth_PRIVATE.json"))
ans = {}
for f in sorted(glob.glob("llm_ex/out_*.json")):
    for r in json.load(open(f)): ans[r["id"]] = r["answer"]
print("answers:", len(ans), "of", len(truth), "| missing:", sorted(set(truth) - set(ans))[:5])
rows = []
for iid, t in truth.items():
    if iid not in ans: continue
    parsed = LX._parse_json(ans[iid] if isinstance(ans[iid], str) else json.dumps(ans[iid]))
    found = bool(parsed and parsed.get("found"))
    lm = pd.Period(t["last_month"], "M")
    sd = 0.3 if t["asset"] != "UNRATE" else 0.15
    try:
        last = float(t["last_val"])
    except Exception:                                   # neg_month items stored the text "YYYY-MM = value"
        last = float(re.search(r"=\s*([0-9.]+)", str(t["last_val"])).group(1))
    new = LX.verify(parsed, t["sent"], t["last_month"], last, sd) if parsed else None
    row = dict(id=iid, type=t["type"], asset=t["asset"], model_found=found, accepted=new is not None)
    if t["type"] == "positive" and new is not None:
        exp = last * (1 + t["value"] / 100.0) if t["kind"] == "pct_change" else t["value"]
        row["correct"] = abs(new - exp) <= (0.0006 * abs(exp) if t["kind"] == "pct_change" else 1e-9)
        row["err"] = new - exp
    if t["type"] == "synthetic" and new is not None and t["value"] is not None:
        exp = last * (1 + t["value"] / 100.0) if t["kind"] == "pct_change" else t["value"]
        row["correct"] = abs(new - exp) <= 1e-9 + (0.0006 * abs(exp) if t["kind"] == "pct_change" else 0)
    rows.append(row)
D = pd.DataFrame(rows)
pos = D[D.type == "positive"]
print(f"\nPOSITIVES (BLS documents, n={len(pos)}):  model said found: {pos.model_found.mean():.0%} | ACCEPTED by verifier: {pos.accepted.mean():.0%}")
acc = pos[pos.accepted]
print(f"   of the accepted: CORRECT {int(acc.correct.sum())}/{len(acc)} = {acc.correct.mean():.1%}   wrong: {int((~acc.correct).sum())}")
print("   accepted rate by asset:", pos.groupby("asset").accepted.mean().round(2).to_dict())
miss = pos[~pos.accepted]; print("   not accepted:", len(miss), "| model found=true but verifier rejected:", int((miss.model_found).sum()))
for k in ("neg_month", "neg_series"):
    n = D[D.type == k]; print(f"\nNEGATIVE CONTROL {k} (n={len(n)}): model said found: {int(n.model_found.sum())} | ACCEPTED by verifier (must be 0): {int(n.accepted.sum())}")
sy = D[D.type == "synthetic"]
print("\nSYNTHETIC foreign formats:"); 
for _, r in sy.iterrows():
    t = truth[r.id]; print(f"   {t['asset']:10s} expected={'(none)' if t['value'] is None else t['value']}  model_found={r.model_found} accepted={r.accepted} correct={r.get('correct')}")
if D[D.type.isin(['neg_month', 'neg_series'])].accepted.sum() or (len(acc) and (~acc.correct).sum()):
    print("\n!!! FALSE ACCEPTS / WRONG ACCEPTED VALUES:"); 
    for _, r in D.iterrows():
        if (r.type.startswith("neg") and r.accepted) or (r.type == "positive" and r.accepted and not r.correct):
            print("  ", r.id, r.type, r.asset, "answer:", str(ans[r.id])[:260], "| err", r.get("err"))
