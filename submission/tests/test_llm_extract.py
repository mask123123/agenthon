"""Verifier + integration invariants for the House-model fallback, with a mocked model (no network)."""
import sys, pathlib, json, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from t2agent import llm_extract as LX, releases as R
from t2agent.house import House

TEXT = ("[release dated 2024-08-14]\nThe Consumer Prices Index (CPI) rose by 0.4% in July 2024 on a seasonally adjusted basis, "
        "compared with a rise of 0.1% in June 2024. The annual rate was 2.2%.")
QUOTE = "The Consumer Prices Index (CPI) rose by 0.4% in July 2024 on a seasonally adjusted basis"
def hist(n=40, last="2024-06-01", val=130.0):
    idx = [str(d.date()) for d in pd.date_range(end=last, periods=n, freq="MS")]
    return pd.Series(np.linspace(val - 8, val, n) + 0.05 * np.sin(np.arange(n)), index=idx)

def good(**kw):
    a = {"found": True, "month": "2024-07", "kind": "pct_change", "value": 0.4, "quote": QUOTE}; a.update(kw); return a

def test_accepts_correct_answer():
    h = hist(); new = LX.verify(good(), TEXT, "2024-06", float(h.iloc[-1]), 0.3)
    assert new is not None and abs(new - float(h.iloc[-1]) * 1.004) < 1e-9

def test_rejects_every_kind_of_bad_answer():
    h = hist(); lv = float(h.iloc[-1])
    bad = [good(quote="The Consumer Prices Index (CPI) rose by 0.5% in July 2024 on a seasonally adjusted basis"),   # invented sentence
           good(value=0.6),                                       # number not in quote
           good(month="2024-08"), good(month="2024-06"),          # wrong month
           good(value=-0.4),                                      # sign contradicts 'rose'
           good(value=12.0, quote="CPI rose by 12.0% in July 2024 and more"),   # implausible
           good(kind="level", value=0.4),                         # level far from last value
           good(quote="rose"), {"found": False}, good(kind="weird")]
    for b in bad:
        assert LX.verify(b, TEXT, "2024-06", lv, 0.3) is None, b

def test_unchanged_and_index_level_rules():
    h = hist(); lv = float(h.iloc[-1])
    t = "The Consumer Price Index was unchanged in July 2024 on a seasonally adjusted basis after rising 0.1 percent in June."
    ok = LX.verify({"found": True, "month": "2024-07", "kind": "pct_change", "value": 0.0, "quote": "The Consumer Price Index was unchanged in July 2024 on a seasonally adjusted basis"}, t, "2024-06", lv, 0.3)
    assert ok is not None and abs(ok - lv) < 1e-12
    assert LX.verify({"found": True, "month": "2024-07", "kind": "pct_change", "value": 0.0, "quote": "after rising 0.1 percent in June"}, t, "2024-06", lv, 0.3) is None   # 0 not supported by the quote
    t2 = "The July 2024 level of 131.0 (1982-84=100) was 2.2 percent higher than in July 2023."   # index level, unadjusted: refused
    assert LX.verify({"found": True, "month": "2024-07", "kind": "level", "value": 131.0, "quote": "The July 2024 level of 131.0 (1982-84=100)"}, t2, "2024-06", 130.7, 0.5) is None
    t3 = "The unemployment rate (not seasonally adjusted) was 4.4 percent in July 2024."
    assert LX.verify({"found": True, "month": "2024-07", "kind": "level", "value": 4.4, "quote": "The unemployment rate (not seasonally adjusted) was 4.4 percent in July 2024"}, t3, "2024-06", 4.1, 0.1) is None

def test_level_kind_and_decline():
    h = hist(); lv = float(h.iloc[-1]); t = "Unemployment: the rate stood at 4.3 percent in July 2024, up from 4.1 percent in June."
    assert abs(LX.verify({"found": True, "month": "2024-07", "kind": "level", "value": 4.3, "quote": "the rate stood at 4.3 percent in July 2024"}, t, "2024-06", 4.1, 0.1) - 4.3) < 1e-9
    t2 = "Prices fell by 0.3% in July 2024 after a rise in June."
    new = LX.verify({"found": True, "month": "2024-07", "kind": "pct_change", "value": -0.3, "quote": "Prices fell by 0.3% in July 2024"}, t2, "2024-06", lv, 0.3)
    assert new is not None and new < lv

class Mock(House):
    def __init__(self, reply, avail=True): super().__init__(); self.reply, self._a, self.calls = reply, avail, 0
    @property
    def available(self): return self._a
    def chat(self, messages, max_tokens=250, timeout_s=45.0):
        self.calls += 1
        if isinstance(self.reply, Exception): raise self.reply
        return self.reply

def _unit(doc_type="macro_release", ts="2024-08-14"):
    d = tempfile.mkdtemp(); p = pathlib.Path(d)
    (p / "ons.txt").write_text(TEXT.split("\n", 1)[1])
    (p / "corpus_index.json").write_text(json.dumps({"documents": [{"doc_id": "x", "timestamp": ts, "doc_type": doc_type, "file": "ons.txt", "source": "ONS"}]}))
    return d

def test_integration_fallback_applies_only_when_verified():
    h = {"CPI_UK": hist()}; ps = np.array([[8.0]]); reply = json.dumps(good())
    h2, ps2, led = R.apply_fresh(h, ps, ["CPI_UK"], _unit(), "2024-08-14", house=Mock(reply))
    assert ps2[0, 0] == 7.0 and str(h2["CPI_UK"].index[-1]) == "2024-07-01" and "House model" in led[0], led
    for r in (json.dumps(good(value=0.9)), "not json", json.dumps({"found": False}), RuntimeError("boom")):
        m = Mock(r); h3, ps3, led3 = R.apply_fresh(h, ps, ["CPI_UK"], _unit(), "2024-08-14", house=m)
        assert ps3 is ps and h3 is h and m.calls == 1, (r, led3)

def test_no_call_without_candidate_or_endpoint_or_when_regex_succeeds():
    h = {"CPI_UK": hist()}; ps = np.array([[8.0]]); m = Mock(json.dumps(good()))
    R.apply_fresh(h, ps, ["CPI_UK"], _unit(doc_type="beige_book"), "2024-08-14", house=m); assert m.calls == 0     # wrong doc type
    R.apply_fresh(h, ps, ["CPI_UK"], _unit(ts="2024-05-01"), "2024-08-14", house=m); assert m.calls == 0          # older than panel
    R.apply_fresh(h, ps, ["CPI_UK"], _unit(), "2024-08-14", house=Mock("x", avail=False)); 
    assert R.apply_fresh(h, ps, ["CPI_UK"], _unit(), "2024-08-14", house=m, use_llm=False)[1] is ps and m.calls == 0

def test_budget_and_breaker():
    h = House(max_requests=1); h.endpoint, h.token, h.model = "http://x", "t", "m"; h.used = 1
    try: h.chat([{"role": "user", "content": "hi"}]); assert False
    except Exception as e: assert "cap" in str(e)
    h2 = House(); h2.fail_streak = 3; h2.endpoint, h2.token, h2.model = "http://x", "t", "m"
    try: h2.chat([{"role": "user", "content": "hi"}]); assert False
    except Exception as e: assert "circuit" in str(e)

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
