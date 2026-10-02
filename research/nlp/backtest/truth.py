"""Offline only. Stitch every unit's panels into full histories and resolve practice-unit outcomes.

Outputs (never packaged into the image, never read by the agent):
  backtest/data/history.parquet   [panel, asset, date, value]   latest-as-of vintage per date
  backtest/data/realized.json     {unit_id: {"cells": [[asset, h, value], ...], "complete": bool, ...}}
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from t2agent.task import load_task, normalize_panel  # noqa: E402

UNITS = ROOT.parents[1] / "track2-forecasting-public" / "units"
OUT = ROOT / "backtest" / "data"


def stitch() -> pd.DataFrame:
    rows = []
    for u in sorted(UNITS.iterdir()):
        card = u / "card.toml"
        if not card.is_file() or not (u / "forecast_spec.json").is_file():
            continue  # the worked exemplar ships synthetic panels; keep it out of real history
        import tomllib

        asof = tomllib.loads(card.read_text())["provenance"]["data_cutoff"]
        for p in sorted(u.glob("*.parquet")) + sorted((u / "panels").glob("*.parquet")):
            df = normalize_panel(pd.read_parquet(p))
            df["panel"] = p.stem
            df["unit_asof"] = asof
            rows.append(df)
    allr = pd.concat(rows, ignore_index=True)
    # consistency of overlapping values across units (vintage differences show up here)
    g = allr.groupby(["panel", "asset", "date"])["value"]
    spread = (g.max() - g.min()).rename("spread").reset_index()
    bad = spread[spread["spread"] > 1e-9]
    report = bad.groupby(["panel", "asset"]).size().rename("n_mismatch_dates").reset_index()
    allr = allr.sort_values("unit_asof").drop_duplicates(["panel", "asset", "date"], keep="last")
    hist = allr[["panel", "asset", "date", "value"]].sort_values(["panel", "asset", "date"]).reset_index(drop=True)
    return hist, report


def resolve(hist: pd.DataFrame) -> dict:
    out = {}
    for u in sorted(UNITS.iterdir()):
        if not (u / "forecast_spec.json").is_file():
            continue
        import tomllib

        asof = tomllib.loads((u / "card.toml").read_text())["provenance"]["data_cutoff"]
        t = load_task(u, u / "text", asof)
        cells, complete, notes = [], True, []
        for a, h in t.cells():
            stem = next((s for s in sorted(t.panels) if (t.panels[s]["asset"] == a).any()), None)
            ser = hist[(hist["panel"] == stem) & (hist["asset"] == a)]
            ser = pd.Series(ser["value"].to_numpy(float), index=pd.to_datetime(ser["date"]))
            if t.observation_periods:
                k = sorted(t.horizons).index(h)
                per = t.observation_periods[t.horizons.index(h)]
                td = pd.Timestamp(per + "-01")
                v = ser.get(td, np.nan)
                notes.append("monthly: latest vintage, approximate")
            else:
                td = pd.Timestamp(str(np.busday_offset(asof, h, roll="backward"))[:10])
                if ser.index.max() < td:
                    v = np.nan
                elif t.target_type == "log_return":
                    r = ser[(ser.index > pd.Timestamp(asof)) & (ser.index <= td)]
                    v = float(np.log1p(r).sum()) if len(r) else np.nan
                else:
                    prev = ser[ser.index <= td]
                    v = float(prev.iloc[-1]) if len(prev) and (td - prev.index[-1]).days <= 5 else np.nan
            if not np.isfinite(v):
                complete = False
            cells.append([a, h, None if not np.isfinite(v) else float(v), str(td.date())])
        out[t.unit_id] = {"asof": asof, "category": t.category, "target_type": t.target_type,
                          "cells": cells, "complete": complete, "notes": sorted(set(notes))}
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    hist, report = stitch()
    hist.to_parquet(OUT / "history.parquet", index=False)
    print("stitched rows:", len(hist))
    print("overlap mismatches (panel, asset, n_dates):")
    print(report.to_string(index=False) if len(report) else "  none")
    real = resolve(hist)
    (OUT / "realized.json").write_text(json.dumps(real, indent=1))
    n_c = sum(v["complete"] for v in real.values())
    print(f"units: {len(real)}  fully resolvable: {n_c}  partial/none: {len(real) - n_c}")
    for k, v in real.items():
        if not v["complete"]:
            print("  unresolved:", k, [c[:2] for c in v["cells"] if c[2] is None])
