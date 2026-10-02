"""Task representation shared by the runtime CLI and the offline backtest.

A Task is everything a forecaster may see for one unit: the card fields that matter, and the
numeric panels truncated at the as-of. The backtest builds Tasks in memory (pseudo-cards); the CLI
builds them from /input. Engines only ever see a Task, so both paths run identical code.
"""

from __future__ import annotations

import json
import pathlib
import tomllib
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Task:
    unit_id: str
    asof: str
    assets: list[str]
    horizons: list[int]
    target_type: str = "level"  # "level" | "log_return"
    target_frequency: str = "daily"
    category: str = ""  # e.g. "T2-F4"; may be empty
    n_draws_min: int = 200
    observation_periods: list[str] | None = None  # aligned with horizons (monthly cards)
    # panel stem -> long frame [date(str), asset, value], rows <= asof
    panels: dict[str, pd.DataFrame] = field(default_factory=dict)
    text_dir: pathlib.Path | None = None
    card: dict[str, Any] = field(default_factory=dict)

    def cells(self) -> list[tuple[str, int]]:
        """Grid order used everywhere: asset id sorted, then horizon ascending (M0's order)."""
        return [(a, h) for a in sorted(self.assets) for h in sorted(self.horizons)]

    def history(self, asset: str) -> pd.Series:
        """The asset's series from the first panel (sorted stem order) that carries it, <= asof."""
        for stem in sorted(self.panels):
            df = self.panels[stem]
            sub = df[df["asset"] == asset]
            if len(sub):
                s = pd.Series(sub["value"].to_numpy(float), index=pd.to_datetime(sub["date"]))
                s = s[~s.index.duplicated(keep="last")].sort_index()
                return s[s.index <= pd.Timestamp(self.asof)].dropna()
        raise KeyError(f"asset {asset!r} not found in any panel")


def normalize_panel(df: pd.DataFrame) -> pd.DataFrame:
    col = "asset" if "asset" in df.columns else ("asset_id" if "asset_id" in df.columns else None)
    if col is None:
        raise ValueError("panel has neither 'asset' nor 'asset_id'")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"),
            "asset": df[col].astype(str),
            "value": pd.to_numeric(df["value"], errors="coerce").astype(float),
        }
    )
    return out.dropna(subset=["value"])


def _find_card(panels_dir: pathlib.Path) -> pathlib.Path:
    for d in (panels_dir, panels_dir.parent, panels_dir.parent.parent):
        if (d / "card.toml").is_file():
            return d / "card.toml"
    raise FileNotFoundError(f"card.toml not found near {panels_dir}")


def load_task(panels_dir: str | pathlib.Path, text_dir: str | pathlib.Path | None, asof: str) -> Task:
    panels_dir = pathlib.Path(panels_dir)
    card_path = _find_card(panels_dir)
    card = tomllib.loads(card_path.read_text())
    spec: dict[str, Any] = {}
    spec_path = card_path.parent / "forecast_spec.json"
    if spec_path.is_file():
        try:
            spec = json.loads(spec_path.read_text())
        except Exception:
            spec = {}

    tg = card.get("targets", {})
    st = spec.get("targets", {}) if isinstance(spec.get("targets"), dict) else {}
    assets = [str(a) for a in (tg.get("asset_ids") or st.get("asset_ids") or [])]
    horizons = [int(h) for h in (tg.get("horizons") or st.get("horizons") or [])]
    if not assets or not horizons:
        raise ValueError("card declares no assets/horizons")
    meta = card.get("metadata", {})
    params = card.get("scoring", {}).get("params", {})

    files = sorted(panels_dir.glob("*.parquet")) or sorted(panels_dir.parent.glob("*.parquet"))
    panels = {}
    for p in files:
        try:
            df = normalize_panel(pd.read_parquet(p))
        except Exception:
            continue
        panels[p.stem] = df[df["date"] <= asof]

    obs = st.get("observation_periods") or tg.get("observation_periods")
    return Task(
        unit_id=str(card.get("task", {}).get("id", "")),
        asof=asof,
        assets=assets,
        horizons=horizons,
        target_type=str(tg.get("target_type") or st.get("target_type") or meta.get("target_type") or "level"),
        target_frequency=str(tg.get("target_frequency") or meta.get("target_frequency") or "daily"),
        category=str(meta.get("category", "")),
        n_draws_min=int(params.get("n_draws_min", 200) or 200),
        observation_periods=[str(x) for x in obs] if obs else None,
        panels=panels,
        text_dir=pathlib.Path(text_dir) if text_dir else None,
        card=card,
    )
