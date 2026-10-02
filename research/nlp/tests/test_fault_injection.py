"""Fault injection for the text layer: every failure must end in 'no overlay', quickly, within budget."""

from __future__ import annotations

import json
import os
import pathlib
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from t2agent.house import House, parse_json  # noqa: E402
from t2agent.task import load_task  # noqa: E402
from t2agent.textlayer import extract, overlay  # noqa: E402

UNIT = ROOT.parents[1] / "track2-forecasting-public" / "units" / "t2-F3-bear-flattener-2022"


def serve(reply_fn):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            code, body = reply_fn()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def run(endpoint: str | None, token: str = "t"):
    for k in ("MODEL_ENDPOINT", "MODEL_TOKEN", "MODEL_NAME", "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(k, None)
    if endpoint:
        os.environ.update(MODEL_ENDPOINT=endpoint, MODEL_TOKEN=token, MODEL_NAME="house")
    t = load_task(UNIT, UNIT / "text", "2022-06-15")
    h = House()
    t0 = time.monotonic()
    res = extract(t, h)
    return res, h.used, time.monotonic() - t0


def chat_reply(content: str) -> str:
    return json.dumps({"choices": [{"message": {"content": content}, "finish_reason": "stop"}], "usage": {}})


def main() -> None:
    cases = []
    cases.append(("no endpoint configured", *run(None)))
    srv, url = serve(lambda: (401, '{"error":"unauthorized"}'))
    cases.append(("401 unauthorized", *run(url)))
    srv.shutdown()
    cases.append(("connection refused", *run("http://127.0.0.1:9")))
    srv, url = serve(lambda: (200, "<html>not json</html>"))
    cases.append(("garbage HTTP body", *run(url)))
    srv.shutdown()
    srv, url = serve(lambda: (200, chat_reply("I cannot help with that.")))
    cases.append(("prose instead of JSON", *run(url)))
    srv.shutdown()
    srv, url = serve(lambda: (200, chat_reply('{"assets": ["UST_2Y"], "policy_actions": "none"}')))
    cases.append(("malformed JSON shape", *run(url)))
    srv.shutdown()

    ok = True
    for name, res, used, secs in cases:
        passed = (res is None or not isinstance((res.get("card") or {}).get("assets"), dict)) and used <= 14 and secs < 120
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'}  {name:26s} result={'None' if res is None else 'dict'} requests={used} {secs:.1f}s")

    draws = np.random.default_rng(0).normal(size=(500, 2))
    cells = [("UST_2Y", 63), ("UST_2Y", 126)]
    for bad in [None, {}, {"assets": []}, {"assets": {"UST_2Y": "x"}},
                {"assets": {"UST_2Y": {"established_action": {"direction": "up", "confidence": "high"}}}},
                {"assets": {"UST_2Y": {"established_action": {"direction": 1, "confidence": 0.9},
                                       "binary_event_in_horizon": {"present": "yes"}, "stress": None}}}]:
        out, _ = overlay(draws, cells, bad)
        good = out.shape == draws.shape and np.isfinite(out).all()
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  overlay on {json.dumps(bad)[:70]}")
    assert parse_json('```json\n{"a": "x\\"}", "b": [1,2,],}\n```') == {"a": 'x"}', "b": [1, 2]}
    print("PASS  parse_json fences/escapes/trailing commas")
    print("ALL PASS" if ok else "SOME FAILED")


if __name__ == "__main__":
    main()
