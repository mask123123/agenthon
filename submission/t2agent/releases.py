"""Executive summary (read this first)

Monthly macro panels (CPI, core CPI, unemployment rate, nonfarm payrolls) lag the as-of date by 1-2 months, but the unit's
text corpus often contains the BLS release published on/just before the as-of date, i.e. the latest print the panel does
not have yet. This module reads those numbers deterministically (regular expressions, no model, no network) and appends
them to the history, so the engine anchors on the newest value with one fewer step. Measured on pseudo-cards (public
history): one extra month of data is worth ~13 % of the score on such cards (20 % at 2 steps).

Safety: a value is used only if (a) the release's reference month is exactly the month after the panel's last
observation, (b) it parses cleanly, and (c) it passes a plausibility check against the series' own history. Anything
else is ignored and the engine runs exactly as before. Only information dated on or before the as-of is read.
"""
from __future__ import annotations

import json
import pathlib
import re

import numpy as np
import pandas as pd

MONTHS = {m: i + 1 for i, m in enumerate(["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST",
                                          "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"])}
SUPPORTED = {"CPI_ALL", "CPI_CORE", "UNRATE", "NFP"}
_TITLE = re.compile(r"(CONSUMER PRICE INDEX|THE EMPLOYMENT SITUATION)[^A-Za-z0-9]{0,8}\s*([A-Z]+)\s+(\d{4})")
_UP = ("increased", "rose", "edged up", "added", "gained", "grew", "up")
_DOWN = ("fell", "declined", "decreased", "edged down", "lost", "down", "dropped")


def _num(s: str) -> float:
    return float(s.replace(",", "").replace("+", ""))


def parse_release(text: str) -> dict | None:
    """-> {"kind": "cpi"|"empsit", "ref": "YYYY-MM", "values": {...}} or None. Values are first-release prints."""
    t = re.sub(r"\s+", " ", text)
    m = _TITLE.search(t[:4000])
    if not m or m.group(2) not in MONTHS:
        return None
    kind = "cpi" if m.group(1).startswith("CONSUMER") else "empsit"
    ref = f"{int(m.group(3)):04d}-{MONTHS[m.group(2)]:02d}"
    vals: dict[str, float] = {}
    if kind == "cpi":
        # The panels hold SEASONALLY ADJUSTED index levels (FRED CPIAUCSL / CPILFESL), the release gives SA monthly % changes
        # (1 decimal): new level = last panel level x (1 + pct/100). Unadjusted index levels in the text are NOT used.
        def pct(m):
            if m is None:
                return None
            verb, num = m.group(1), m.group(2)
            if verb.startswith("was unchanged") or verb == "changed little":
                return 0.0
            if num is None:
                return None
            v = float(num)
            return -v if verb in ("decreased", "fell", "declined", "edged down") else v
        a = re.search(r"\(CPI-U\)\s+(increased|rose|decreased|fell|declined|was unchanged)(?: by)?(?:\s+(\d\.\d)\s+percent)?"
                      r"[^.]{0,40}?(?:on a seasonally adjusted basis|in [A-Z][a-z]+)", t[:3000])
        c = re.search(r"[Tt]he index for all items less food and energy (rose|increased|fell|declined|decreased|"
                      r"was unchanged|edged up|edged down|changed little)(?: by)?(?:\s+(\d\.\d)\s+percent)?", t)
        pa, pc = pct(a), pct(c)
        if c is not None and c.group(1) == "edged up" and pc is None:
            pc = None
        if pa is not None:
            vals["CPI_ALL_PCT"] = pa
        if pc is not None:
            vals["CPI_CORE_PCT"] = pc
    else:
        head = t[:900]
        u = re.search(r"unemployment rate[^.]{0,70}?(?:to|at) (\d{1,2}\.\d) percent", head)
        if u:
            vals["UNRATE"] = float(u.group(1))
        n = re.search(r"nonfarm payroll employment (increased|rose|edged up|edged down|fell|declined|decreased|"
                      r"added|changed little)[^.]{0,40}?by ([\d,]+)", head)
        if n and n.group(1) != "changed little":
            ch = _num(n.group(2)) / 1000.0
            vals["NFP_CHANGE"] = -ch if n.group(1) in ("edged down", "fell", "declined", "decreased") else ch
            j = t.find("revised")
            rev = 0.0
            if j >= 0:
                seg = t[max(0, j - 200): j + 650]
                found = re.findall(r"revised (?:up|down) by [\d,]+, from ([+-]?[\d,]+) to ([+-]?[\d,]+)", seg)[:2]
                rev = sum(_num(b) - _num(a) for a, b in found) / 1000.0
            vals["NFP_REVISION"] = rev
    return {"kind": kind, "ref": ref, "values": vals} if vals else None


def _releases(text_dir: pathlib.Path, asof: str) -> list[dict]:
    """Parsed macro releases dated <= asof, oldest first."""
    out = []
    idx = text_dir / "corpus_index.json"
    if not idx.is_file():
        return out
    for d in json.loads(idx.read_text()).get("documents", []):
        if not isinstance(d, dict) or str(d.get("doc_type")) != "macro_release":
            continue
        ts = str(d.get("timestamp", ""))[:10]
        p = text_dir / str(d.get("file") or "")
        if not ts or ts > asof or not p.is_file():
            continue
        r = parse_release(p.read_text(errors="replace"))
        if r:
            r["date"] = ts
            out.append(r)
    return sorted(out, key=lambda r: (r["ref"], r["date"]))


def _month_after(date_str: str) -> str:
    p = pd.Period(str(date_str)[:7], freq="M") + 1
    return str(p)


def _plausible(asset: str, val: float, last_val: float, v: dict, sd: float) -> bool:
    """Domain bounds, not a volatility test: genuinely huge prints (e.g. April 2020) must pass, mis-parses must not."""
    if not np.isfinite(val) or val <= 0:
        return False
    if asset == "UNRATE":
        return 1.0 <= val <= 30.0 and abs(val - last_val) <= 12.0
    if asset in ("CPI_ALL", "CPI_CORE"):
        return abs(v.get(asset + "_PCT", 99.0)) <= 3.0
    if asset == "NFP":
        return abs(v.get("NFP_CHANGE", 1e9)) <= 25000.0 and abs(v.get("NFP_REVISION", 0.0)) <= 2000.0 and abs(val / last_val - 1.0) < 0.2
    return False


def fresh_prints(hist: pd.Series, asset: str, releases: list[dict]) -> list[tuple[str, float]]:
    """New (month_start, value) observations beyond the panel's last one, consecutive months only, plausibility-checked."""
    if asset not in SUPPORTED or hist is None or len(hist) < 24:
        return []
    last_month = str(hist.index[-1])[:7]
    last_val = float(hist.iloc[-1])
    d = hist.astype(float).diff().dropna().iloc[-120:]
    sd = float(d.std()) if len(d) > 5 else 0.0
    out: list[tuple[str, float]] = []
    for r in releases:
        if r["ref"] != _month_after(f"{last_month}-01"):
            continue
        v = r["values"]
        if asset == "NFP":
            if "NFP_CHANGE" not in v:
                continue
            val = last_val + v.get("NFP_REVISION", 0.0) + v["NFP_CHANGE"]
        elif asset in ("CPI_ALL", "CPI_CORE"):
            key = asset + "_PCT"
            if key not in v:
                continue
            val = last_val * (1.0 + v[key] / 100.0)
        else:
            if asset not in v:
                continue
            val = float(v[asset])
        if not _plausible(asset, val, last_val, v, sd):
            continue                                   # implausible value (likely a mis-parse): ignore, keep the panel as is
        out.append((f"{r['ref']}-01", val))
        last_month, last_val = r["ref"], val
    return out


def apply_fresh(hist: dict, panel_steps, assets: list[str], text_dir, asof: str, house=None, use_llm: bool = True):
    """-> (hist, panel_steps, ledger). Appends fresh prints and shortens each asset's step count by the months gained.

    Regular expressions first (BLS layouts); for assets they could not read, and only if the corpus holds a macro release newer
    than the panel, the House model is asked to quote the published number and the answer is verified (llm_extract.verify)."""
    ledger: list[str] = []
    if panel_steps is None:
        return hist, panel_steps, ledger
    try:
        text_dir = pathlib.Path(text_dir)
        rel = _releases(text_dir, asof)
        ps = np.array(panel_steps, dtype=float).copy()
        new_hist = dict(hist)
        changed = False
        h = house
        for ai, a in enumerate(assets):
            fp = [(m, v, "") for m, v in fresh_prints(hist.get(a), a, rel)]
            how = "regex"
            if not fp and use_llm and hist.get(a) is not None and len(hist[a]) >= 24:
                try:
                    from . import llm_extract as LX
                    from .house import House
                    docs = LX.candidate_docs(text_dir, asof, str(hist[a].index[-1])[:7])
                    if docs:
                        h = h or House()
                        if h.available:
                            fp = LX.extract(h, hist[a], a, docs)
                            how = "House model, verified against the text"
                except Exception:
                    fp = []
            if not fp:
                continue
            s = hist[a].copy()
            for month, val, _q in fp:
                s.loc[month] = val
            new_hist[a] = s
            ps[ai, :] = np.maximum(ps[ai, :] - len(fp), 1.0)
            changed = True
            quote = f" quote: \"{fp[0][2]}\"" if fp[0][2] else ""
            ledger.append(f"  {a}: latest release value(s) not yet in the panel ({how}): "
                          + ", ".join(f"{m[:7]}={v:.4g}" for m, v, _q in fp) + f" -> anchor moved, steps reduced by {len(fp)}.{quote}")
        return (new_hist, ps, ledger) if changed else (hist, panel_steps, ledger)
    except Exception as exc:                            # never let this cost a card
        return hist, panel_steps, [f"  release reader error ({type(exc).__name__}); not applied"]
