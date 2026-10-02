"""Release reader invariants: parses known releases, only consecutive newer months, plausibility guard, no-op otherwise."""
import sys, pathlib, json, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from t2agent import releases as R

CPI = "CONSUMER PRICE INDEX - JUNE 2023 The Consumer Price Index for All Urban Consumers (CPI-U) rose 0.2 percent in June on a seasonally adjusted basis, after increasing 0.1 percent in May. The index for all items less food and energy rose 0.4 percent in June."
EMP = "THE EMPLOYMENT SITUATION -- JULY 2024 The unemployment rate rose to 4.3 percent in July, and nonfarm payroll employment edged up by 114,000, the U.S. Bureau reported. The change in total nonfarm payroll employment for May was revised down by 2,000, from +218,000 to +216,000, and the change for June was revised down by 27,000, from +206,000 to +179,000."
OLD_STYLE = "CONSUMER PRICE INDEX MAY 2017 The Consumer Price Index for All Urban Consumers (CPI-U) decreased 0.1 percent in May on a seasonally adjusted basis, the U.S. Bureau of Labor Statistics reported today."

def series(vals, start="2020-01-01"):
    idx = [str(d.date()) for d in pd.date_range(start, periods=len(vals), freq="MS")]
    return pd.Series(vals, index=idx, dtype=float)

def test_parse():
    c = R.parse_release(CPI); assert c["ref"] == "2023-06" and c["values"]["CPI_ALL_PCT"] == 0.2 and c["values"]["CPI_CORE_PCT"] == 0.4, c
    e = R.parse_release(EMP); assert e["ref"] == "2024-07" and e["values"]["UNRATE"] == 4.3 and e["values"]["NFP_CHANGE"] == 114.0 and e["values"]["NFP_REVISION"] == -29.0, e
    o = R.parse_release(OLD_STYLE); assert o["ref"] == "2017-05" and o["values"]["CPI_ALL_PCT"] == -0.1, o
    assert R.parse_release("not a release") is None

def test_consecutive_and_values():
    s = series(list(np.linspace(290, 303, 54)), "2019-01-01")           # ends 2023-06-01 -> release for 2023-06 is NOT newer
    rel = [R.parse_release(CPI) | {"date": "2023-07-12"}]
    assert R.fresh_prints(s, "CPI_ALL", rel) == []
    s2 = s.iloc[:-1]                                                      # ends 2023-05-01 -> consecutive
    fp = R.fresh_prints(s2, "CPI_ALL", rel); assert len(fp) == 1 and fp[0][0] == "2023-06-01" and abs(fp[0][1] - float(s2.iloc[-1]) * 1.002) < 1e-9
    u = series([4.0 + 0.01 * (i % 5) for i in range(60)], "2019-07-01"); u = u.iloc[:-1] if str(u.index[-1]) >= "2024-07-01" else u
    rel_e = [R.parse_release(EMP) | {"date": "2024-08-02"}]
    s3 = series([4.1] * 40, "2021-03-01")                                  # ends 2024-06-01
    assert str(s3.index[-1]) == "2024-06-01"
    assert R.fresh_prints(s3, "UNRATE", rel_e) == [("2024-07-01", 4.3)]
    n = series(list(150000 + np.arange(40) * 120.0), "2021-03-01")
    fn = R.fresh_prints(n, "NFP", rel_e); assert fn and abs(fn[0][1] - (n.iloc[-1] - 29 + 114)) < 1e-9

def test_plausibility_guard():
    s = series([4.1 + 0.01 * (i % 3) for i in range(40)], "2021-03-01")   # tiny monthly sd
    for badv in (0.3, 99.0):                                                # mis-parsed numbers are ignored
        bad = [{"kind": "empsit", "ref": "2024-07", "date": "2024-08-02", "values": {"UNRATE": badv}}]
        assert R.fresh_prints(s, "UNRATE", bad) == []
    big = [{"kind": "empsit", "ref": "2024-07", "date": "2024-08-02", "values": {"UNRATE": 14.7}}]   # a real April-2020-style print
    assert R.fresh_prints(s, "UNRATE", big) == [("2024-07-01", 14.7)]
    cpi_bad = [{"kind": "cpi", "ref": "2024-07", "date": "2024-08-14", "values": {"CPI_ALL_PCT": 9.9}}]
    assert R.fresh_prints(series([300 + i * 0.3 for i in range(40)], "2021-03-01"), "CPI_ALL", cpi_bad) == []

def test_apply_noop_without_releases():
    h = {"UNRATE": series([4.1] * 40, "2021-03-01")}; ps = np.array([[8.0, 9.0]])
    with tempfile.TemporaryDirectory() as d:
        h2, ps2, led = R.apply_fresh(h, ps, ["UNRATE"], d, "2024-08-02")
    assert ps2 is ps and led == [] and h2 is h
    h3, ps3, led3 = R.apply_fresh(h, None, ["UNRATE"], "/nonexistent", "2024-08-02"); assert ps3 is None

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
