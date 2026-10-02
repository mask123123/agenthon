"""Executive summary (read this first)

Fallback for `releases.py`: when the corpus holds a macro release newer than the lagging monthly panel but the regular
expressions did not read it (other country, other publisher, other layout), the House model is asked ONE narrow question -
"what latest value does this release publish for this series?" - and must answer with a number plus the verbatim sentence it
came from. The model never forecasts and never judges. Code then verifies the answer before anything is used:

  1. the quote occurs verbatim in the text we sent (whitespace / typographic-quote tolerant);
  2. the value occurs as a number inside that quote (sign must agree with up/down wording);
  3. the stated month is exactly the month after the panel's last observation and its name appears in the text;
  4. the value is plausible (level only for rate-like series and never unadjusted, within 20 % and 8 monthly s.d.;
     monthly % change within +-10 %; a stated 'unchanged' counts as 0).

Anything that fails is discarded (the engine runs exactly as without this module). At most one request per asset and
`House.max_requests` per unit; any exception means "no answer". Kinds: "level" (index / rate level in the panel's units)
and "pct_change" (monthly % change, preferably seasonally adjusted, applied to the panel's last value).
"""
from __future__ import annotations

import calendar
import json
import pathlib
import re

import numpy as np
import pandas as pd

from .house import House

MAX_CHARS_PER_DOC = 3500
MAX_DOCS = 2

SYS = ("You extract published numbers from official statistical releases. Answer with a single JSON object and nothing else. "
       "Never estimate, infer or recall; only report what the text states.")
USER = """A forecaster tracks the monthly series "{asset}". Its last known observation is {last_month} = {last_val:g}.
Below are the opening parts of statistical release(s) published after that. Find the latest value this release publishes for
{asset} for the month {want_month} (the month right after the last observation).

Return JSON: {{"found": true|false, "month": "YYYY-MM", "kind": "level"|"pct_change", "value": <number>, "quote": "<exact sentence copied from the text>"}}
Rules: "level" = the rate level stated in the text (use the seasonally adjusted headline figure when both exist); "pct_change" = the one-month percent change (use the seasonally adjusted one when both exist; negative if it fell; 0 if the text says unchanged).
The quote must be copied character for character from the text and must contain the number. If the text does not state it, return {{"found": false}}.

TEXT:
{text}"""

_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "})
_UP = ("rose", "increased", "up", "gained", "grew", "higher", "rise", "increase", "climbed", "edged up")
_DOWN = ("fell", "decreased", "down", "declined", "lower", "drop", "dropped", "decline", "decrease", "edged down")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.translate(_QUOTES)).strip().lower()


def candidate_docs(text_dir: pathlib.Path, asof: str, last_month: str) -> list[tuple[str, str]]:
    """Newest macro_release documents dated after the panel's last observation month and on/before the as-of."""
    idx = text_dir / "corpus_index.json"
    if not idx.is_file():
        return []
    start = str(pd.Period(last_month, freq="M") + 1) + "-01"
    docs = []
    try:
        for d in json.loads(idx.read_text()).get("documents", []):
            if isinstance(d, dict) and str(d.get("doc_type")) == "macro_release":
                ts = str(d.get("timestamp", ""))[:10]
                p = text_dir / str(d.get("file") or "")
                if start <= ts <= asof and p.is_file():
                    docs.append((ts, p))
    except Exception:
        return []
    docs.sort(reverse=True)
    out = []
    for ts, p in docs[:MAX_DOCS]:
        out.append((ts, re.sub(r"\s+", " ", p.read_text(errors="replace")).strip()[:MAX_CHARS_PER_DOC]))
    return out


def _parse_json(s: str) -> dict | None:
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def verify(ans: dict, sent_text: str, last_month: str, last_val: float, sd: float) -> float | None:
    """Return the new level if `ans` passes every check, else None."""
    try:
        if not ans.get("found"):
            return None
        want = str(pd.Period(last_month, freq="M") + 1)
        if str(ans.get("month")) != want:
            return None
        kind, value, quote = ans.get("kind"), float(ans.get("value")), str(ans.get("quote", ""))
        if kind not in ("level", "pct_change") or not np.isfinite(value) or len(quote) < 12:
            return None
        nt, nq = _norm(sent_text), _norm(quote)
        if nq not in nt:
            return None                                                    # quote not verbatim in what we sent
        nums = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", nq) if re.search(r"\d", x)]
        zero_ok = kind == "pct_change" and value == 0.0 and re.search(r"\b(unchanged|no change|did not change|was flat)\b", nq)
        if not zero_ok and not any(abs(abs(value) - n) < 1e-9 * max(1.0, n) for n in nums):
            return None                                                    # number is not in the quote
        words = set(re.findall(r"[a-z]+", nq))
        up, down = bool(words & set(_UP)), bool(words & set(_DOWN))
        if kind == "pct_change":
            if value < 0 and not down:
                return None
            if value > 0 and down and not up:
                return None
        mon = calendar.month_name[int(want[5:7])].lower()
        if mon not in nt and mon[:3] + " " not in nt and mon[:3] + "." not in nt:
            return None                                                    # reference month not named in the text
        new = last_val * (1.0 + value / 100.0) if kind == "pct_change" else value
        if kind == "pct_change" and abs(value) > 10.0:
            return None
        if kind == "level":
            # Index levels are published unadjusted while panels hold seasonally adjusted values (a ~1 % mismatch passes any
            # volatility test), so only rate-like series (|value| < 30, e.g. a % rate) may use a stated level.
            if last_val <= 0 or last_val >= 30.0 or re.search(r"not seasonally adjusted|unadjusted|non-seasonally|nsa\b", nq):
                return None
            if not (0.8 <= new / last_val <= 1.25):
                return None
            if sd > 1e-9 * abs(last_val) and abs(new - last_val) > 8.0 * sd:
                return None
        return float(new) if np.isfinite(new) and new > 0 else None
    except Exception:
        return None


def extract(house: House, hist: pd.Series, asset: str, docs: list[tuple[str, str]]) -> list[tuple[str, float, str]]:
    """-> [(month_start, value, quote)] with at most one verified new observation; [] on any problem."""
    if not docs or hist is None or len(hist) < 24:
        return []
    try:
        last_month = str(hist.index[-1])[:7]
        last_val = float(hist.iloc[-1])
        d = hist.astype(float).diff().dropna().iloc[-120:]
        sd = float(d.std()) if len(d) > 5 else 0.0
        want = str(pd.Period(last_month, freq="M") + 1)
        text = "\n\n".join(f"[release dated {ts}]\n{t}" for ts, t in docs)
        out = house.chat([{"role": "system", "content": SYS},
                          {"role": "user", "content": USER.format(asset=asset, last_month=last_month, last_val=last_val,
                                                                  want_month=want, text=text)}], max_tokens=300)
        ans = _parse_json(out)
        new = verify(ans, text, last_month, last_val, sd) if ans else None
        return [(f"{want}-01", new, str(ans.get("quote", ""))[:160])] if new is not None else []
    except Exception:
        return []
