"""Ratio-aware resampling invariants.  Run from submission/:  python tests/test_ratio_aware.py"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from t2agent import ratio_aware as RA  # noqa: E402


def _cards(shape):
    rng = np.random.default_rng(3)
    n, a, h = shape
    m0 = rng.normal(0, 1, (500, a, h))
    eng = rng.normal(0.3, 1.2, (n, a, h))  # engine belief: shifted and wider than M0
    return eng, m0


def test_crps_matches_official_estimator():
    from qfbench2_common.scoring import crps
    rng = np.random.default_rng(0)
    m0, x = rng.normal(size=(500, 3)), rng.normal(size=(9, 3)) * 2
    ref = np.array([crps.crps_ensemble(m0, row) for row in x])
    assert np.abs(RA.m0_crps_at(m0, x) - ref).max() < 1e-12


def test_rows_are_whole_engine_draws_and_shape_kept():
    eng, m0 = _cards((1000, 2, 2))
    out, ledger = RA.resample(eng, m0, "unit-a")
    assert out.shape == eng.shape and np.isfinite(out).all()
    flat = {tuple(r) for r in eng.reshape(len(eng), -1).round(12)}
    assert all(tuple(r) in flat for r in out.reshape(len(out), -1).round(12)), "rows must be whole engine draws"
    assert "gamma 0.25" in ledger[0] and "multi" in ledger[0]


def test_deterministic_per_unit():
    eng, m0 = _cards((1000, 1, 1))
    a, _ = RA.resample(eng, m0, "unit-b")
    b, _ = RA.resample(eng, m0, "unit-b")
    assert np.array_equal(a, b)


def test_moves_mass_toward_where_m0_scores_well():
    eng, m0 = _cards((2000, 1, 1))
    out, ledger = RA.resample(eng, m0, "unit-c")
    before = RA.m0_crps_at(m0.reshape(500, -1), eng.reshape(len(eng), -1)).mean()
    after = RA.m0_crps_at(m0.reshape(500, -1), out.reshape(len(out), -1)).mean()
    assert after < before and "gamma 0.1" in ledger[0] and "single" in ledger[0]


def test_unusable_m0_leaves_draws_unchanged():
    eng, m0 = _cards((500, 2, 1))
    for bad in (m0[:, :1, :], np.full_like(m0, np.nan)):
        out, ledger = RA.resample(eng, bad, "unit-d")
        assert np.array_equal(out, eng) and "not applied" in ledger[0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
