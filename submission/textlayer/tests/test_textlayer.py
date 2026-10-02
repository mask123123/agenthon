"""Text-layer tests: every failure must leave the draws unchanged; success must only shift UST cells, within the cap.

Run from submission/:  PYTHONPATH=$PWD python -m pytest textlayer/tests -q   (or: python textlayer/tests/test_textlayer.py)
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from textlayer import text_overlay  # noqa: E402
from textlayer.fed_tone import find_statements  # noqa: E402
from textlayer.house import House  # noqa: E402

STATEMENT = ("The Federal Open Market Committee decided today to raise the target range for the federal funds rate "
             "and anticipates that ongoing increases in the target range will be appropriate.")


def _serve(reply):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            code, payload = reply(body)
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload.encode())

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _logprob_reply(body):
    q = body["messages"][-1]["content"]
    top = ([("4", 0.7), ("5", 0.2), ("3", 0.1)] if "single digit" in q else [("A", 0.8), ("B", 0.15), ("C", 0.05)])
    choice = {"message": {"content": top[0][0]},
              "logprobs": {"content": [{"token": top[0][0], "logprob": math.log(top[0][1]),
                                        "top_logprobs": [{"token": t, "logprob": math.log(p)} for t, p in top]}]}}
    return 200, json.dumps({"choices": [choice]})


def _unit(tmp: pathlib.Path, n_statements: int = 2) -> pathlib.Path:
    text = tmp / "text"
    text.mkdir(parents=True, exist_ok=True)
    docs = []
    for i, d in enumerate(["2022-05-04", "2022-06-15"][:n_statements]):
        (text / f"fomc_statement_{d}.txt").write_text(STATEMENT + f" ({i})")
        docs.append({"doc_id": f"s{i}", "timestamp": d, "doc_type": "fomc_statement", "file": f"fomc_statement_{d}.txt"})
    (text / "speech.txt").write_text("unrelated speech")
    docs.append({"doc_id": "sp", "timestamp": "2022-06-10", "doc_type": "cb_speech", "file": "speech.txt"})
    (text / "corpus_index.json").write_text(json.dumps({"documents": docs}))
    return text


def _inputs():
    rng = np.random.default_rng(0)
    assets, horizons = ["UST_2Y", "EUR"], [63, 126]
    samples = rng.normal(size=(1000, 2, 2)) + np.array([3.0, 1.1])[None, :, None]
    idx = pd.bdate_range("2021-01-01", "2022-06-15")
    hist = {"UST_2Y": pd.Series(np.linspace(0.1, 3.0, len(idx)), index=idx),
            "EUR": pd.Series(np.linspace(1.2, 1.05, len(idx)), index=idx)}
    return samples, assets, horizons, hist


def _env(endpoint):
    for k in ("MODEL_ENDPOINT", "MODEL_TOKEN", "MODEL_NAME", "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(k, None)
    if endpoint:
        os.environ.update(MODEL_ENDPOINT=endpoint, MODEL_TOKEN="t", MODEL_NAME="house")


def _run(endpoint, n_statements=2):
    _env(endpoint)
    samples, assets, horizons, hist = _inputs()
    with tempfile.TemporaryDirectory() as d:
        out, ledger = text_overlay(samples, assets, horizons, hist, _unit(pathlib.Path(d), n_statements), "2022-06-15")
    return samples, out, ledger


def _check_shift(before, after, max_sd=0.100001):
    assert np.array_equal(before[:, 1, :], after[:, 1, :]), "non-UST cells must not move"
    shift_sd = (after[:, 0, :].mean(0) - before[:, 0, :].mean(0)) / before[:, 0, :].std(0)
    assert np.all(np.abs(shift_sd) <= max_sd) and np.any(shift_sd != 0), shift_sd
    assert np.allclose(after[:, 0, :].std(0), before[:, 0, :].std(0)), "width must not change"
    assert np.isfinite(after).all()


def test_house_failure_falls_back_to_lexicon():
    cases = {"no endpoint": lambda: _run(None), "connection refused": lambda: _run("http://127.0.0.1:9")}
    servers = []
    for name, reply in {"401": lambda b: (401, "{}"), "garbage": lambda b: (200, "<html>"),
                        "prose": lambda b: (200, json.dumps({"choices": [{"message": {"content": "I think hawkish"}}]}))}.items():
        srv, url = _serve(reply)
        servers.append(srv)
        cases[name] = (lambda u: (lambda: _run(u)))(url)
    for name, fn in cases.items():
        before, after, ledger = fn()
        _check_shift(before, after)
        assert any("lexicon model" in line for line in ledger), (name, ledger)
    for srv in servers:
        srv.shutdown()


def test_no_statement_unchanged():
    srv, url = _serve(_logprob_reply)
    before, after, ledger = _run(url, n_statements=0)
    srv.shutdown()
    assert np.array_equal(before, after) and "no FOMC statement" in ledger[0]


def test_sentiment_is_reported_not_applied():
    srv, url = _serve(_logprob_reply)
    _, after_a, ledger = _run(url)
    srv.shutdown()
    assert any("sentiment (reported only)" in line for line in ledger)
    import textlayer.lexicon as lx
    saved = lx.econ_sentiment, lx.uncertainty
    lx.econ_sentiment, lx.uncertainty = (lambda t: 9.0), (lambda t: 99.0)  # wildly different sentiment
    srv, url = _serve(_logprob_reply)
    _, after_b, _ = _run(url)
    srv.shutdown()
    lx.econ_sentiment, lx.uncertainty = saved
    assert np.array_equal(after_a, after_b), "sentiment must not move the forecast"


def test_success_shifts_only_ust_within_cap():
    srv, url = _serve(_logprob_reply)
    before, after, ledger = _run(url)
    srv.shutdown()
    _check_shift(before, after)
    assert any("model 'house'" in line for line in ledger), ledger


def test_no_logprobs_falls_back_to_answer_token():
    srv, url = _serve(lambda b: (200, json.dumps({"choices": [{"message": {"content": "4" if "single digit" in b["messages"][-1]["content"] else "B"}}]})))
    before, after, ledger = _run(url)
    srv.shutdown()
    assert any("hawkishness 4.00" in line for line in ledger), ledger


def test_budget_never_exceeds_cap():
    calls = []
    srv, url = _serve(lambda b: (calls.append(1), (500, "{}"))[1])
    _run(url)
    srv.shutdown()
    assert len(calls) <= 6


def test_prompts_match_the_ones_coefs_were_fitted_with():
    from textlayer import fed_tone
    research = pathlib.Path(__file__).resolve().parents[3] / "research" / "nlp" / "backtest" / "nemotron_tone.py"
    if not research.is_file():  # image build context has no research/ tree
        return
    src = research.read_text()
    for s in (fed_tone.SYS, fed_tone.Q1, fed_tone.Q2):
        assert s in src, "prompt drifted from the fitted one: refit coefs.json or revert the prompt"


def test_lexicons_match_the_ones_coefs_were_fitted_with():
    import ast
    from textlayer import lexicon
    base = pathlib.Path(__file__).resolve().parents[3] / "research" / "nlp" / "backtest"
    if not base.is_dir():
        return
    found = {}
    for f in ("step1_nlp.py", "step1b_compare.py"):
        for node in ast.parse((base / f).read_text()).body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                found[node.targets[0].id] = ast.literal_eval(node.value) if isinstance(node.value, ast.List) else None
    for name in ("HAWK", "DOVE", "ECON_POS", "ECON_NEG", "UNCERT"):
        assert getattr(lexicon, name) == found[name], f"{name} drifted from the fitted lexicon"


def test_find_statements_respects_asof():
    with tempfile.TemporaryDirectory() as d:
        t = _unit(pathlib.Path(d))
        assert [x[0] for x in find_statements(t, "2022-06-15")] == ["2022-05-04", "2022-06-15"]
        assert [x[0] for x in find_statements(t, "2022-06-01")] == ["2022-05-04"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
