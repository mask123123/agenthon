"""End-to-end: real CLI + real HTTP against a local mock of the House route (/v1/chat/completions, Bearer token).
The unit is a copy of the Sahm-watch card whose BLS release is rewritten so that the regular expressions cannot read it."""
import sys, os, json, shutil, subprocess, tempfile, threading, http.server, re, pathlib
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
UNIT = "/Users/apple/Documents/agenthon/track2-forecasting-public/units/t2-F1-sahm-watch-2024"
QUOTE = "the jobless rate stood at 4.3 percent in July 2024"
DOC = ("Labour market report. In the household survey the jobless rate stood at 4.3 percent in July 2024, up from 4.1 percent in June. "
       "Employers added 114,000 positions over the month.")

def make_unit():
    d = tempfile.mkdtemp(); u = pathlib.Path(d) / "unit"; shutil.copytree(UNIT, u)
    for f in (u / "text").glob("empsit_2024-08-02*"):
        f.write_text(DOC)                                      # no 'THE EMPLOYMENT SITUATION' title -> regex returns nothing
    return u

class H(http.server.BaseHTTPRequestHandler):
    mode = "good"; seen = []
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); body = json.loads(self.rfile.read(n))
        H.seen.append({"path": self.path, "auth": self.headers.get("Authorization"), "think": body.get("chat_template_kwargs"), "max_tokens": body.get("max_tokens")})
        if H.mode == "500": self.send_response(500); self.end_headers(); return
        ans = {"good": json.dumps({"found": True, "month": "2024-07", "kind": "level", "value": 4.3, "quote": QUOTE}),
               "fabricated": json.dumps({"found": True, "month": "2024-07", "kind": "level", "value": 4.3, "quote": "unemployment rose to 4.3 percent in July 2024"}),
               "wrongvalue": json.dumps({"found": True, "month": "2024-07", "kind": "level", "value": 4.4, "quote": QUOTE}),
               "garbage": "I think the rate is about four percent."}.get(H.mode, "")
        out = json.dumps({"choices": [{"message": {"content": ans}}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)

def run(unit, endpoint=None):
    out = tempfile.mkdtemp(); env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("HTTP_PROXY", "HTTPS_PROXY"))}
    env.update(NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost", PYTHONPATH=str(ROOT))
    if endpoint: env.update(MODEL_ENDPOINT=endpoint, MODEL_TOKEN="tok-123", MODEL_NAME="house")
    r = subprocess.run([sys.executable, "-m", "t2agent.cli", "--panels", str(unit) + "/", "--text", str(unit) + "/text/", "--asof", "2024-08-02",
                        "--out", out + "/f.parquet", "--card", str(unit) + "/card.toml"], capture_output=True, text=True, cwd="/tmp", env=env, timeout=120)
    assert r.returncode == 0, r.stderr[-400:]
    f = pd.read_parquet(out + "/f.parquet"); med = float(np.median(f[f.horizon == f.horizon.min()].value))
    return med, open(out + "/forecast_rationale.md").read()

if __name__ == "__main__":
    srv = http.server.HTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
    ep = f"http://127.0.0.1:{srv.server_port}"; u = make_unit()
    base, rat0 = run(u)                                         # no endpoint
    print(f"no endpoint            : median {base:.3f} (anchor stays 4.1)  | text ledger present: {'House model' in rat0}")
    assert base < 4.15 and "House model" not in rat0
    for mode in ("good", "fabricated", "wrongvalue", "garbage", "500"):
        H.mode, H.seen = mode, []
        med, rat = run(u, ep)
        used = "House model, verified" in rat
        print(f"mode {mode:11s}       : median {med:.3f} | applied: {used} | requests: {len(H.seen)} | path {H.seen[0]['path'] if H.seen else '-'}")
        if mode == "good":
            assert used and med > 4.25 and len(H.seen) == 1
            assert H.seen[0]["auth"] == "Bearer tok-123" and H.seen[0]["think"] == {"enable_thinking": False} and H.seen[0]["path"] == "/v1/chat/completions"
            assert QUOTE in rat
        else:
            assert not used and abs(med - base) < 1e-9 and len(H.seen) == 1, (mode, med, base)
    print("PASS end-to-end (good answer applied; fabricated / wrong value / garbage / 500 all leave the forecast exactly unchanged)")
