"""Fallback ladder: an engine failure goes to the exact M0 replica (scores ~1.0), not straight to the Gaussian walk.

Run from submission/:  python tests/test_m0_fallback.py
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from t2agent import cli  # noqa: E402


def _hist(n=400):
    idx = pd.bdate_range("2020-01-01", periods=n).strftime("%Y-%m-%d")
    rng = np.random.default_rng(1)
    return {"UST_2Y": pd.Series(2 + np.cumsum(rng.normal(0, 0.03, n)), index=idx),
            "UST_10Y": pd.Series(3 + np.cumsum(rng.normal(0, 0.03, n)), index=idx)}


def test_engine_failure_uses_m0():
    res = cli._fallback(_hist(), ["UST_2Y", "UST_10Y"], [21, 63], "level", "unit-x", 1000, None, None)
    assert res.meta.get("m0_fallback") and res.samples.shape == (1000, 2, 2) and np.isfinite(res.samples).all()
    sd = res.samples.std(0)
    assert sd[0, 1] > sd[0, 0], "longer horizon must be wider (path structure)"


def test_m0_failure_uses_gaussian():
    res = cli._fallback({"CPI_ALL": pd.Series([300.0], index=["2023-05-01"])}, ["CPI_ALL"], [140, 160], "level",
                        "unit-y", 500, np.array([[8, 9]]), ["2024-01", "2024-02"])
    assert res.meta.get("fallback") and np.isfinite(res.samples).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
