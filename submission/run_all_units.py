"""Run our CLI over every public unit and check the official gates. Prints failures + fallback count."""
import glob, os, subprocess, sys, json, pathlib, tempfile
units = sorted(glob.glob("/Users/apple/Documents/agenthon/track2-forecasting-public/units/t2-*"))
bad, fb = [], []
for u in units:
    out = tempfile.mkdtemp(prefix="t2o_")
    card = json.load(open(u + "/forecast_spec.json")) if os.path.exists(u + "/forecast_spec.json") else {}
    asof = [l for l in open(u + "/card.toml") if l.startswith("data_cutoff")][0].split('"')[1]
    panels = u + "/panels" if os.path.isdir(u + "/panels") else u
    r = subprocess.run([sys.executable, "-m", "t2agent.cli", "--panels", panels + "/", "--text", u + "/text/", "--asof", asof,
                        "--out", out + "/forecast.parquet", "--card", u + "/card.toml"], capture_output=True, text=True)
    if r.returncode != 0:
        bad.append((os.path.basename(u), "CLI", r.stderr[-300:])); continue
    if "[FALLBACK]" in r.stdout: fb.append(os.path.basename(u))
    s = subprocess.run([sys.executable, "/Users/apple/Documents/agenthon/track2-forecasting-public/scoring/scoring.py", "score",
                        "--card", u + "/card.toml", "--forecast", out + "/forecast.parquet"], capture_output=True, text=True)
    try:
        j = json.loads(s.stdout)
        if not j.get("admissible"): bad.append((os.path.basename(u), "GATE", json.dumps(j.get("gates"))[:200]))
    except Exception:
        bad.append((os.path.basename(u), "SCORER", (s.stdout + s.stderr)[-300:]))
print("units", len(units), "failures", len(bad), "fallbacks", len(fb))
for b in bad: print(b)
print("fallback units:", fb)
