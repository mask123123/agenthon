"""Nemotron structural-fact layer (schema v1, frozen in RESEARCH_LOG.md 2026-10-02).

The model only extracts; every number that touches the forecast is set here in code, small and capped.
Any failure returns None and the caller keeps the pure engine draws.
"""

from __future__ import annotations

import json
import pathlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .house import House, parse_json
from .task import Task

MAX_DOCS = 10
HEAD, TAIL = 24_000, 6_000
SHIFT_K, SHIFT_MIN_CONF = 0.10, 0.70
EVENT_W, CRISIS_W, WIDTH_CAP = 0.08, 0.08, 1.15

SYSTEM = (
    "You are a careful financial analyst. You extract facts from documents dated on or before an as-of date. "
    "Use only what the documents state. Never use knowledge of events after the as-of date. "
    "Reply with a single JSON object and nothing else."
)

MAP_PROMPT = """As-of date: {asof}
Target series: {targets}
Document ({doc_type}, dated {date}):
<<<
{body}
>>>

Extract only facts this document STATES that bear on the target series over the next 1-9 months. JSON:
{{"policy_actions": [{{"what": "...", "direction_for": {{"<asset>": -1|1}}, "quote": "<= 25 words"}}],
  "commitments": [{{"what": "forward guidance or stated plan", "direction_for": {{"<asset>": -1|1}}, "quote": "..."}}],
  "scheduled_events": [{{"what": "decision/vote/election/review", "when": "date or text", "quote": "..."}}],
  "regime_constraints": [{{"type": "peg|floor|ceiling|band|cap", "asset_or_pair": "...", "level": null, "quote": "..."}}],
  "stress_signals": [{{"what": "...", "quote": "..."}}]}}
Direction convention: +1 means the target series (yield level, or FX quoted as units per USD as given, or factor
return) would be pushed UP. Leave a list empty if the document states nothing relevant. Do not speculate."""

REDUCE_PROMPT = """As-of date: {asof}
Card family: {family}. Target type: {ttype}. Targets (asset, horizon in business days): {cells}
Recent numbers (per asset): {numbers}
Facts extracted from the dated documents (one object per document, newest first):
{facts}

For EACH target asset decide, from the facts only:
- established_action: a decided or formally announced action/commitment that pushes the series one way over the
  horizon. confidence = probability the series ends the horizon on that side of its last value; use >= 0.7 only
  when the facts are explicit and point the same way. direction 0 if nothing is established.
- binary_event_in_horizon: true only if a scheduled discrete decision, vote, election or policy review falls inside
  the horizon and could move the series sharply either way.
- stress: "calm", "elevated" or "crisis" as the documents describe conditions.
- regime_constraint: an explicit peg/floor/ceiling/band/cap on this exact series, else type "none".
Think briefly. Reply JSON only:
{{"assets": {{"<asset>": {{"established_action": {{"direction": -1|0|1, "confidence": 0.5, "what": "...",
"evidence": "doc date + quote"}}, "binary_event_in_horizon": {{"present": false, "what": "", "evidence": ""}},
"stress": "calm", "regime_constraint": {{"type": "none", "level": null, "evidence": ""}}}}}}}}"""


def load_docs(text_dir: pathlib.Path) -> list[dict]:
    idx_path = text_dir / "corpus_index.json"
    docs = []
    if idx_path.is_file():
        try:
            idx = json.loads(idx_path.read_text())
            docs = [d for d in idx.get("documents", []) if isinstance(d, dict)]
        except Exception:
            docs = []
    if not docs:  # no usable index: every text file, undated
        docs = [{"file": p.name, "timestamp": "", "doc_type": "document"}
                for p in sorted(text_dir.glob("*")) if p.is_file() and p.name != "corpus_index.json"]
    out = []
    for d in sorted(docs, key=lambda d: str(d.get("timestamp", "")), reverse=True):
        f = text_dir / str(d.get("file") or d.get("path") or "")
        if f.is_file():
            body = f.read_text(errors="replace")
            if len(body) > HEAD + TAIL:
                body = body[:HEAD] + "\n[...]\n" + body[-TAIL:]
            out.append({"date": str(d.get("timestamp", ""))[:10], "doc_type": str(d.get("doc_type", "")), "body": body})
        if len(out) >= MAX_DOCS:
            break
    return out


def numbers(task: Task) -> dict:
    out = {}
    for a in sorted(task.assets):
        s = task.history(a)
        x = (np.log1p(s) if task.target_type == "log_return" else s.diff()).dropna()
        r = {"daily_sd_21d": round(float(x.iloc[-21:].std()), 5), "daily_sd_252d": round(float(x.iloc[-252:].std()), 5)}
        if task.target_type == "log_return":
            r["cum_log_return_63d"] = round(float(x.iloc[-63:].sum()), 4)
        else:
            r["last"] = round(float(s.iloc[-1]), 4)
            r["change_63d"] = round(float(s.iloc[-1] - s.iloc[-64]), 4) if len(s) > 64 else None
        out[a] = r
    return out


def _call_json(house: House, prompt: str, *, thinking: bool, max_tokens: int):
    for _ in range(2):  # one retry, charged
        try:
            got = parse_json(house.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                                        thinking=thinking, max_tokens=max_tokens,
                                        timeout_s=240 if thinking else 90))
            if isinstance(got, dict):
                return got
        except Exception as e:
            if type(e).__name__ in ("BudgetExhausted", "TimeoutError"):
                return None
    return None


def extract(task: Task, house: House, parallel: int = 4) -> dict | None:
    if not house.available or task.text_dir is None or not task.text_dir.is_dir():
        return None
    docs = load_docs(task.text_dir)
    if not docs:
        return None
    targets = ", ".join(sorted(task.assets))
    prompts = [MAP_PROMPT.format(asof=task.asof, targets=targets, doc_type=d["doc_type"], date=d["date"], body=d["body"])
               for d in docs]
    with ThreadPoolExecutor(parallel) as ex:
        facts = list(ex.map(lambda p: _call_json(house, p, thinking=False, max_tokens=1200), prompts))
    facts = [dict(f, date=d["date"], doc_type=d["doc_type"]) for f, d in zip(facts, docs) if f]
    if not facts:
        return None
    red = REDUCE_PROMPT.format(asof=task.asof, family=task.category or "unknown", ttype=task.target_type,
                               cells=task.cells(), numbers=json.dumps(numbers(task)),
                               facts=json.dumps(facts, ensure_ascii=False)[:60_000])
    card = _call_json(house, red, thinking=True, max_tokens=4000)
    if not card or not isinstance(card.get("assets"), dict):
        return None
    return {"card": card, "facts": facts}


def overlay(draws: np.ndarray, cells: list[tuple[str, int]], card: dict | None) -> tuple[np.ndarray, list[str]]:
    """Apply the frozen v1 mapping. Returns new draws and a ledger of what was applied."""
    if not card:
        return draws, ["no text overlay (extraction unavailable)"]
    assets = card.get("assets", {}) if isinstance(card.get("assets"), dict) else {}
    out, ledger = draws.copy(), []
    for i, (a, h) in enumerate(cells):
        v = assets.get(a) or {}
        if not isinstance(v, dict):
            continue
        col = out[:, i]
        sd, med = float(np.std(col)), float(np.median(col))
        w = 1.0
        ev = v.get("binary_event_in_horizon") or {}
        if isinstance(ev, dict) and ev.get("present") is True:
            w += EVENT_W
        if str(v.get("stress", "")).lower() == "crisis":
            w += CRISIS_W
        w = min(w, WIDTH_CAP)
        shift = 0.0
        act = v.get("established_action") or {}
        try:
            d, p = int(act.get("direction", 0)), float(act.get("confidence", 0.5))
        except (TypeError, ValueError):
            d, p = 0, 0.5
        if d in (-1, 1) and p >= SHIFT_MIN_CONF:
            shift = SHIFT_K * sd * d
        if w != 1.0 or shift:
            out[:, i] = med + w * (col - med) + shift
            ledger.append(f"{a}@{h}: width x{w:.2f}, shift {shift:+.4g} ({act.get('what', '')[:80]})")
    return out, ledger or ["text overlay: no adjustment triggered"]
