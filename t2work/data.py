"""Executive summary: load the longest public copy of each T2 panel (long format -> wide DataFrames).
Panels come from the public practice units (tuning on practice data is permitted by the README)."""
from __future__ import annotations
import glob, os
import pandas as pd

REPO = "/Users/apple/Documents/agenthon/track2-forecasting-public"
PANELS = {"rates": "rates_daily.parquet", "fx": "g10_fx_daily.parquet", "factors": "factors_daily.parquet",
          "macro": "macro_monthly.parquet"}


def load_wide(name: str) -> pd.DataFrame:
    """Wide date x asset frame from the unit carrying the latest end date for this panel."""
    best, best_end = None, None
    for f in glob.glob(f"{REPO}/units/*/{PANELS[name]}") + glob.glob(f"{REPO}/units/*/panels/{PANELS[name]}"):
        d = pd.read_parquet(f)
        end = d["date"].max()
        if best_end is None or end > best_end:
            best, best_end = d, end
    best["date"] = pd.to_datetime(best["date"])
    return best.pivot(index="date", columns="asset", values="value").sort_index()


if __name__ == "__main__":
    for n in PANELS:
        w = load_wide(n)
        print(n, w.shape, w.index.min().date(), w.index.max().date(), "NaN%", round(w.isna().mean().mean() * 100, 2))
