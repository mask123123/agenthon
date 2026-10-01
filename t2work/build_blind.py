"""Build anonymised, blind labelling items. For each practice card: the 1-2 newest documents (age <= 7 days), type-specific
informative excerpts, with dates/years/months/person names/institution-specific identifiers masked. NO card id, family, asset,
horizon, price or outcome is included. A private key file maps item ids back to card/as-of for later analysis."""
import json, glob, os, re, random, tomllib
from datetime import date
U = "/Users/apple/Documents/agenthon/track2-forecasting-public/units"
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
random.seed(7)

def clean(t):
    t = re.sub(r"^# source:[^\n]*\n", "", t)
    t = re.sub(r"https?://\S+", "", t)
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{2,}", "\n", t)).strip()

def excerpt(typ, t):
    t = clean(t)
    if typ == "fomc_minutes":
        out = []
        for key, n in (("Participants’ Views on Current Conditions", 2600), ("Participants' Views on Current Conditions", 2600), ("Committee Policy Action", 1800)):
            i = t.find(key)
            if i >= 0: out.append(t[i:i + n])
        return "\n[...]\n".join(out) if out else t[:4000]
    if typ == "beige_book":   return t[:3500]
    if typ == "macro_release": return t[:2800]
    if typ == "fomc_statement": return re.sub(r"(?s)Voting for the FOMC monetary policy action were.*", "", t)[:3800]
    if typ == "landmark":      return t[:3500]
    return t[:4500]

def anonymise(t):
    t = re.sub(r"\b(19|20)\d{2}\b", "[YEAR]", t)
    t = re.sub(rf"\b({MONTHS})\.?\s+\d{{1,2}}(,?\s+\[YEAR\])?", "[DATE]", t)
    t = re.sub(rf"\b({MONTHS})\b", "[MONTH]", t)
    t = re.sub(r"\b(Mr|Ms|Mrs|Dr|Professor|Governor|Chairman|Chair|Chairwoman|President)\.?\s+([A-Z][a-z]+\s+){0,2}[A-Z][a-z]+", r"[OFFICIAL]", t)
    t = re.sub(r"\b[A-Z][a-z]+ [A-Z]\.? [A-Z][a-z]+\b(?=:)", "[OFFICIAL]", t)
    return t

items, key = [], {}
for u in sorted(glob.glob(U + "/t2-*")):
    p = u + "/text/corpus_index.json"
    if not os.path.exists(p): continue
    c = tomllib.load(open(u + "/card.toml", "rb")); asof = date.fromisoformat(c["provenance"]["data_cutoff"])
    docs = json.load(open(p))["documents"]
    cand = [((asof - date.fromisoformat(d["timestamp"])).days, d) for d in docs]; cand.sort(key=lambda x: x[0])
    cand = [(a, d) for a, d in cand if a <= 7][:2]
    if not cand: cand = [min((((asof - date.fromisoformat(d["timestamp"])).days, d) for d in docs), key=lambda x: x[0])]
    parts = []
    for age, d in cand:
        txt = open(u + "/text/" + d["file"], errors="ignore").read()
        parts.append(f"[DOCUMENT TYPE: {d['doc_type']}; AGE: {age} days before the forecast date]\n" + anonymise(excerpt(d["doc_type"], txt)))
    iid = "I%03d" % random.randint(0, 999)
    while iid in key: iid = "I%03d" % random.randint(0, 999)
    items.append({"id": iid, "text": "\n\n".join(parts)}); key[iid] = {"unit": os.path.basename(u), "asof": str(asof), "family": c["metadata"]["category"], "year": asof.year}
random.shuffle(items)
for b in range(0, len(items), 13):
    json.dump(items[b:b + 13], open(f"blind/batch_{b // 13:02d}.json", "w"), ensure_ascii=False, indent=1)
json.dump(key, open("blind_key_PRIVATE.json", "w"), indent=1)
print("items", len(items), "batches", (len(items) + 12) // 13, "| avg chars/item", sum(len(i["text"]) for i in items) // len(items))
print("\n----- SAMPLE ITEM (first 1800 chars) -----\n", items[0]["text"][:1800])
