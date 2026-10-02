"""Build blind reading packets for the Opus text-edge test (two arms).

Arm A ("named"): real dates and text, numeric summary with levels.
Arm B ("masked"): years/dates/months, well-known officials and signature event words masked,
                  document dates given relative to the as-of, no levels (changes and vols only).
Both arms: packet ids are anonymous (unit ids name the episode), the author's card prose and
`[text].notes` are withheld, documents are renamed D01.. and capped in length.
The packet -> unit mapping is written to backtest/data/opus_map.json (never shown to the reader).
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from t2agent.task import load_task  # noqa: E402

UNITS = ROOT.parents[1] / "track2-forecasting-public" / "units"
DATA = ROOT / "backtest" / "data"
OUT = ROOT / "backtest" / "opus_packets"
PER_DOC, PER_CARD = 4000, 14000

NAMES = ["Greenspan", "Bernanke", "Yellen", "Powell", "Trichet", "Draghi", "Lagarde", "Duisenberg",
         "Kuroda", "Shirakawa", "Ueda", "Fukui", "Carney", "Bailey", "King", "Stevens", "Lowe", "Bullock",
         "Hildebrand", "Jordan", "Poloz", "Macklem", "Carney", "Wheeler", "Orr", "Ingves", "Thedeen",
         "Olsen", "Wolden Bache", "Nakaso", "Hu", "Zhou Xiaochuan", "Yi Gang", "Rajan", "Tombini",
         "Trump", "Obama", "Biden", "Bush", "Clinton", "Harris", "Romney", "Lehman", "Bear Stearns",
         "Silicon Valley Bank", "Silicon Valley", "SVB", "SIVB", "Signature Bank", "First Republic", "SBNY", "FRC", "Credit Suisse", "Merkel", "Tsipras", "Cameron", "May", "Johnson",
         "Dudley", "Fischer", "Clarida", "Brainard", "Williams", "Bullard", "Kocherlakota", "Evans",
         "Plosser", "Lacker", "Fisher", "Hoenig", "Rosengren", "Waller", "Jefferson", "Vice Chair"]
EVENTS = ["COVID-19", "COVID", "coronavirus", "Coronavirus", "pandemic", "Pandemic", "Brexit", "Grexit",
          "whatever it takes", "Whatever it takes", "fiscal cliff", "Fiscal Cliff", "taper tantrum",
          "Taper Tantrum", "debt ceiling", "Debt Ceiling", "Omicron", "Delta variant", "Ukraine", "Russia",
          "September 11", "9/11", "Katrina", "Fukushima", "Tohoku", "Hurricane", "Evergrande"]
MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|December|"
          "Jan\\.|Feb\\.|Mar\\.|Apr\\.|Aug\\.|Sept\\.|Sep\\.|Oct\\.|Nov\\.|Dec\\.")


def mask(text: str) -> str:
    text = re.sub(r"\b(19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b", "[DATE]", text)
    text = re.sub(r"\b\d{1,2}/\d{1,2}/(19|20)?\d{2}\b", "[DATE]", text)
    text = re.sub(rf"\b({MONTHS})\b", "[MONTH]", text)
    text = re.sub(r"\b(19|20)\d{2}s?\b", "[YEAR]", text)
    text = re.sub(r"\b(Q[1-4]|H[12])\s*\[YEAR\]", "[PERIOD]", text)
    for w in sorted(set(EVENTS), key=len, reverse=True):
        text = text.replace(w, "[EVENT]")
    for n in sorted(set(NAMES), key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(n)}\b", "[PERSON]", text)
    return text


def numeric_summary(t, masked: bool) -> list[dict]:
    rows = []
    for a in sorted(t.assets):
        s = t.history(a)
        x = np.log1p(s) if t.target_type == "log_return" else s.diff()
        x = x.dropna()
        r = {"asset": a}
        if t.target_type == "log_return":
            r["note"] = "panel rows are daily simple returns; target is the cumulative log return over the horizon"
            for k in (21, 63, 252):
                r[f"cum_log_return_last_{k}d"] = round(float(x.iloc[-k:].sum()), 4)
        else:
            if not masked:
                r["last_value"] = round(float(s.iloc[-1]), 4)
            for k in (21, 63, 252):
                r[f"change_last_{k}d"] = round(float(s.iloc[-1] - s.iloc[-min(k, len(s) - 1) - 1]), 4)
        r["daily_sd_last_21d"] = round(float(x.iloc[-21:].std()), 5)
        r["daily_sd_last_252d"] = round(float(x.iloc[-252:].std()), 5)
        rows.append(r)
    return rows


def build() -> None:
    real = json.loads((DATA / "realized.json").read_text())
    ids = sorted(u for u, r in real.items() if r["complete"])
    rng = np.random.default_rng(2026)
    order = rng.permutation(len(ids))
    mapping = {}
    for arm in ("A", "B"):
        (OUT / arm).mkdir(parents=True, exist_ok=True)
    for k, i in enumerate(order):
        uid = ids[i]
        pid = f"P{k + 1:03d}"
        mapping[pid] = uid
        u = UNITS / uid
        t = load_task(u, u / "text", real[uid]["asof"])
        idx = json.loads((u / "text" / "corpus_index.json").read_text())
        docs = sorted(idx.get("documents", []), key=lambda d: d.get("timestamp", ""), reverse=True)
        for arm in ("A", "B"):
            masked = arm == "B"
            d = OUT / arm / pid
            d.mkdir(parents=True, exist_ok=True)
            budget = PER_CARD
            listing = []
            for j, doc in enumerate(docs):
                f = u / "text" / doc.get("file", "")
                if not f.is_file() or budget <= 0:
                    continue
                body = f.read_text(errors="replace")[: min(PER_DOC, budget)]
                budget -= len(body)
                ts = doc.get("timestamp", "")[:10]
                when = f"T-{(pd.Timestamp(t.asof) - pd.Timestamp(ts)).days} calendar days" if masked else ts
                name = f"D{j + 1:02d}.txt"
                head = f"doc_type: {doc.get('doc_type', '')}\ndate: {when}\n\n"
                (d / name).write_text(head + (mask(body) if masked else body))
                listing.append({"doc": name, "doc_type": doc.get("doc_type", ""), "date": when})
            spec = {
                "packet_id": pid,
                "asof": "[hidden]" if masked else t.asof,
                "card_family": t.category,
                "target_type": t.target_type,
                "targets": [{"asset": a, "horizon_business_days": h} for a, h in t.cells()],
                "numeric_summary": numeric_summary(t, masked),
                "documents": listing,
            }
            (d / "spec.json").write_text(json.dumps(spec, indent=1))
    (DATA / "opus_map.json").write_text(json.dumps(mapping, indent=1))
    print("packets:", len(mapping))


if __name__ == "__main__":
    build()
