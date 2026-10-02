"""Fed tone from the unit's own corpus: the latest FOMC statement dated on or before the as-of, and the one before it.

n1 = expected value of the answer digit (1 very dovish .. 5 very hawkish) under the House model's first-token
probabilities; n2 = P(more hawkish) - P(more dovish) when shown the previous and the current statement.
The prompts are byte-identical to the ones the coefficients in coefs.json were fitted with — do not edit them
without refitting.
"""

from __future__ import annotations

import json
import pathlib
import re
import time

from .house import House

MAX_CHARS = 8000

SYS = "You are an expert on Federal Reserve communication. Judge only the text you are given. Answer with one character."
Q1 = """FOMC statement:
<<<
{s}
>>>
On a scale from 1 to 5, how hawkish is this statement's monetary policy stance and guidance?
1 = very dovish (easing, strong accommodation), 3 = neutral, 5 = very hawkish (tightening, inflation-fighting).
Answer with a single digit."""
Q2 = """Previous FOMC statement:
<<<
{p}
>>>
Current FOMC statement:
<<<
{s}
>>>
Compared with the previous statement, is the current statement's policy stance and guidance
A = more hawkish, B = about the same, C = more dovish? Answer with a single letter."""


def find_statements(text_dir: pathlib.Path, asof: str) -> list[tuple[str, str]]:
    """[(date, text), ...] of FOMC statements in the corpus dated <= asof, oldest first."""
    docs = []
    idx = text_dir / "corpus_index.json"
    if idx.is_file():
        try:
            for d in json.loads(idx.read_text()).get("documents", []):
                if not isinstance(d, dict):
                    continue
                name = str(d.get("file") or d.get("path") or "")
                if str(d.get("doc_type", "")) == "fomc_statement" or "fomc_statement" in name:
                    docs.append((str(d.get("timestamp", ""))[:10], text_dir / name))
        except Exception:
            docs = []
    if not docs:  # no usable index: fall back to file names
        for p in text_dir.glob("*fomc_statement*"):
            m = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", p.name)
            if m:
                docs.append((f"{m.group(1)}-{m.group(2)}-{m.group(3)}", p))
    out = []
    for date, path in sorted(docs):
        if date and date <= asof and path.is_file():
            text = re.sub(r"\s+", " ", path.read_text(errors="replace")).strip()
            if text:
                out.append((date, text[:MAX_CHARS]))
    return out


def _msg(q: str) -> list[dict]:
    return [{"role": "system", "content": SYS}, {"role": "user", "content": q}]


def tone(house: House, latest: str, previous: str | None) -> tuple[float | None, float | None]:
    n1 = n2 = None
    try:
        p = house.first_token_probs(_msg(Q1.format(s=latest)))
    except Exception:  # one retry on a transient upstream error; it costs one more request slot
        time.sleep(2)
        p = house.first_token_probs(_msg(Q1.format(s=latest)))
    mass = sum(p.get(str(i), 0.0) for i in range(1, 6))
    if mass > 0.5:
        n1 = sum(i * p.get(str(i), 0.0) for i in range(1, 6)) / mass
    if previous:
        try:
            p = house.first_token_probs(_msg(Q2.format(p=previous, s=latest)))
            mass = p.get("A", 0.0) + p.get("B", 0.0) + p.get("C", 0.0)
            if mass > 0.5:
                n2 = (p.get("A", 0.0) - p.get("C", 0.0)) / mass
        except Exception:
            n2 = None
    return n1, n2
