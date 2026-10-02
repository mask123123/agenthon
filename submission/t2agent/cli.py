"""Executive summary (read this first)

The `forecast` verb: reads the unit (card.toml, panels, text dir), builds joint draws with
`engine.build_draws`, and writes forecast.parquet + forecast_meta.json + forecast_rationale.md.
A failed card costs 4.0 (worst case), so every stage degrades instead of raising: engine ->
simple Gaussian-walk fallback -> still writes valid files. Output files are mode 0644, dirs 0755
(the output checker reads them as another user). The text dir is only searched for BLS macro releases (numbers), never for judgement.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tomllib
import traceback

import numpy as np
import pandas as pd

from qfbench2_track_forecasting.cli import _read_panels, _series, _monthly_inputs
from qfbench2_track_forecasting.limits import ParseLimits

from . import engine

DEFAULT_DRAWS = 1000
F4_DRAWS = 2000


def _context_logvol(panels, targets, asof):
    """Median daily log-return std (last 300 obs) over the OTHER daily, positive-valued series in the panels.
    Used only for transfer-style targets. Returns None when fewer than 3 usable series exist."""
    vols = []
    for df in panels.values():
        col = "asset" if "asset" in df.columns else ("asset_id" if "asset_id" in df.columns else None)
        if col is None or "value" not in df.columns:
            continue
        for name, g in df.groupby(col):
            if str(name) in targets:
                continue
            g = g.copy(); g["date"] = pd.to_datetime(g["date"].astype(str).str.slice(0, 10))
            g = g[g["date"] <= pd.Timestamp(asof)].sort_values("date")
            v = g["value"].astype(float)
            if len(v) < 200 or (v <= 0).any() or g["date"].diff().dt.days.median() > 3:
                continue
            vols.append(float(np.log(v).diff().dropna().iloc[-300:].std()))
    return float(np.median(vols)) if len(vols) >= 3 else None


def _find_card(panels_dir: pathlib.Path, explicit: pathlib.Path | None) -> pathlib.Path:
    if explicit and explicit.exists():
        return explicit
    for cand in (panels_dir / "card.toml", panels_dir.parent / "card.toml", panels_dir.parent.parent / "card.toml"):
        if cand.exists():
            return cand
    raise SystemExit(f"card.toml not found near {panels_dir}; pass --card")


def _rationale(unit_id, asof, assets, horizons, family, n_draws, samples, res, note, ledger=None) -> str:
    m = res.meta
    rows = []
    for ai, a in enumerate(assets):
        for hi, h in enumerate(horizons):
            q = np.quantile(samples[:, ai, hi], [0.05, 0.5, 0.95])
            rows.append(f"| {a} | {h} | {m.get('last', {}).get(a, float('nan')):.5g} | "
                        f"{q[0]:.5g} | {q[1]:.5g} | {q[2]:.5g} |")
    body = "\n".join(rows)
    text_part = ("Only numbers: the corpus was searched for BLS releases newer than the lagging panel; values read from them "
                 "(no judgement, no model):\n" + "\n".join(ledger)) if ledger else \
                "Nothing: no document was read for judgement; every adjustment above is statistical."
    if m.get("fallback"):
        method = ("Fallback Gaussian random walk from the supplied history only (the main engine could not "
                  f"run: {note}).")
    else:
        method = (f"Random walk anchored at the last observation (zero for log-return targets), drift = "
                  f"{m['drift_mult']} x the sample mean step, spread from the trailing-window step volatility "
                  f"scaled by sqrt(steps), width multiplier {m['k']:.2f}, multivariate Student-t({m['nu']:.0f}) "
                  "shocks with one shared mixing variable so assets and horizons move together. "
                  f"Transfer-style target (data hole in history): {m['transfer']}. Monthly steps: {m['monthly']}.")
    return f"""# Forecast rationale - {unit_id}

As of **{asof}**; {len(assets)} asset(s) x {len(horizons)} horizon key(s) {horizons}; {n_draws} joint draws; card family {family or 'n/a'}.

## Method
{method}
Only panel rows dated at or before the as-of were used. The card family and declared horizons are the only
card metadata read.

## Adjustment ledger (summary of the output distribution)
| asset | horizon key | anchor | 5% | median | 95% |
|---|---|---|---|---|---|
{body}

## What the text corpus contributed
{text_part}

## What would change this
Different volatility in the trailing window, or any document-based evidence (not used here).
"""


def _write(out_path: pathlib.Path, unit_id, asof, assets, horizons, n_draws, samples, target, rationale):
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    d_idx, a_idx, h_idx = np.meshgrid(np.arange(n_draws), np.arange(len(assets)), np.arange(len(horizons)), indexing="ij")
    df = pd.DataFrame({
        "draw": d_idx.ravel().astype("int32"),
        "asset": np.array(assets, dtype=object)[a_idx.ravel()],
        "horizon": np.array(horizons, dtype="int32")[h_idx.ravel()],
        "value": samples[d_idx.ravel(), a_idx.ravel(), h_idx.ravel()].astype("float64"),
    })
    df.to_parquet(out_path, index=False)
    (out_dir / "forecast_meta.json").write_text(json.dumps({
        "unit_id": unit_id, "asof": asof, "representation": "samples", "asset_ids": assets,
        "horizons": horizons, "n_draws": n_draws, "target": target,
        "rationale": {"file": "forecast_rationale.md", "method": "shrunk-drift t-walk, statistical only"},
    }, indent=2) + "\n")
    (out_dir / "forecast_rationale.md").write_text(rationale if rationale.strip() else "# Forecast rationale\nstatistical forecast\n")
    try:                                    # the checker reads as another user
        os.chmod(out_dir, 0o755)
        for p in out_dir.iterdir():
            os.chmod(p, 0o644 if p.is_file() else 0o755)
    except OSError:
        pass


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="forecast")
    p.add_argument("--panels", type=pathlib.Path, required=True)
    p.add_argument("--text", type=pathlib.Path, required=True)
    p.add_argument("--asof", required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--card", type=pathlib.Path, default=None)
    p.add_argument("--n-draws", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)         # accepted, unused: seed derives from unit id
    a = p.parse_args(argv)

    card_path = _find_card(a.panels, a.card)
    card = tomllib.loads(card_path.read_text())
    tgt = card.get("targets", {})
    assets = list(tgt["asset_ids"])
    horizons = [int(h) for h in tgt["horizons"]]
    target_type = tgt.get("target_type", "level")
    unit_id = card.get("task", {}).get("id", "unknown-unit")
    family = str(card.get("metadata", {}).get("category", ""))
    floor = int(card.get("scoring", {}).get("params", {}).get("n_draws_min", 0) or 0)
    lim = ParseLimits()
    n_draws = max(a.n_draws or (F4_DRAWS if family == "T2-F4" else DEFAULT_DRAWS), floor, lim.min_draws)
    n_draws = min(n_draws, lim.max_draws)

    note = ""
    hist: dict = {}
    res = None
    panel_steps = None
    text_ledger: list[str] = []
    try:
        panels = _read_panels(a.panels)
        for asset in assets:
            try:
                hist[asset] = _series(panels, asset, a.asof)
            except (SystemExit, Exception):
                pass
        try:
            panel_steps = _monthly_inputs(panels, card, card_path, a.asof)
        except (SystemExit, Exception):
            panel_steps = None
        if panel_steps is not None and all(k in hist for k in assets):
            from . import releases              # latest BLS print in the corpus that the lagging panel does not have yet
            hist, panel_steps, text_ledger = releases.apply_fresh({k: hist[k] for k in assets}, panel_steps, assets, a.text, a.asof)
        res = engine.build_draws({k: hist[k] for k in assets}, assets, horizons, target_type, unit_id,
                                 n_draws, family=family, panel_steps=panel_steps,
                                 context_logvol=_context_logvol(panels, assets, a.asof))
    except (SystemExit, Exception) as exc:                # never crash: fall back
        note = f"{type(exc).__name__}: {exc}"
        print("main engine failed -> fallback:", note, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        res = engine.fallback_draws(hist, assets, horizons, target_type, unit_id, n_draws, panel_steps=panel_steps)

    samples = res.samples
    if not np.isfinite(samples).all():                    # last safety net before writing
        res = engine.fallback_draws(hist, assets, horizons, target_type, unit_id, n_draws, panel_steps=panel_steps)
        samples = np.nan_to_num(res.samples, nan=0.0, posinf=0.0, neginf=0.0)
    rat = _rationale(unit_id, a.asof, assets, horizons, family, n_draws, samples, res, note, text_ledger)
    _write(a.out, unit_id, a.asof, assets, horizons, n_draws, samples, target_type, rat)
    print(f"wrote forecast for {unit_id}: {len(assets)}x{len(horizons)} cells, {n_draws} draws"
          + (" [FALLBACK]" if res.meta.get("fallback") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
