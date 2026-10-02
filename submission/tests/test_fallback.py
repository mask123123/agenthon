"""Fallback ladder invariants: never raises, finite, and a monthly macro card's fallback width is the same order of
magnitude as the main engine's (it used to be ~5x too wide because the business-day key was used as a step count)."""
import sys, pathlib, glob, tomllib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from t2agent import engine

U = "/Users/apple/Documents/agenthon/track2-forecasting-public/units"

def test_monthly_fallback_width():
    u = U + "/t2-F1-cpi-glidepath-2023"; c = tomllib.load(open(u + "/card.toml", "rb"))
    p = pd.read_parquet(u + "/macro_monthly.parquet"); a = c["targets"]["asset_ids"][0]; hz = c["targets"]["horizons"]
    s = p[p.asset == a].sort_values("date").set_index("date").value; s.index = pd.to_datetime(s.index)
    steps = np.array([[8, 9]], float)
    main = engine.build_draws({a: s}, [a], hz, "level", "t", 2000, panel_steps=steps).samples[:, 0, :]
    for ps in (steps, None):                    # explicit monthly steps, and the spacing-rule path
        fb = engine.fallback_draws({a: s}, [a], hz, "level", "t", 2000, panel_steps=ps).samples[:, 0, :]
        assert np.isfinite(fb).all()
        for i in range(len(hz)):
            wm = np.quantile(main[:, i], .95) - np.quantile(main[:, i], .05); wf = np.quantile(fb[:, i], .95) - np.quantile(fb[:, i], .05)
            assert 0.5 < wf / wm < 2.5, (ps is None, i, wf / wm)

def test_daily_fallback_unchanged_scale():
    u = U + "/t2-EXAMPLE-ust-curve-1m"; p = pd.read_parquet(u + "/rates_daily.parquet")
    s = p[p.asset == "UST_10Y"].sort_values("date").set_index("date").value; s.index = pd.to_datetime(s.index)
    fb = engine.fallback_draws({"UST_10Y": s}, ["UST_10Y"], [21], "level", "t", 2000).samples.reshape(-1)
    main = engine.build_draws({"UST_10Y": s}, ["UST_10Y"], [21], "level", "t", 2000).samples.reshape(-1)
    assert np.isfinite(fb).all() and 0.5 < fb.std() / main.std() < 2.0

def test_never_raises_on_empty():
    r = engine.fallback_draws({}, ["X"], [21, 63], "level", "t", 500)
    assert r.samples.shape == (500, 1, 2) and np.isfinite(r.samples).all()

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
