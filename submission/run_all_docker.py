"""Run every public unit through the built image under platform-like limits; check official gates."""
import glob, os, subprocess, sys, json, tempfile, shutil
units = sorted(glob.glob("/Users/apple/Documents/agenthon/track2-forecasting-public/units/t2-*"))
SC = "/Users/apple/Documents/agenthon/track2-forecasting-public/scoring/scoring.py"
bad, fb = [], []
for u in units:
    out = tempfile.mkdtemp(prefix="t2d_"); os.chmod(out, 0o777)
    asof = [l for l in open(u + "/card.toml") if l.startswith("data_cutoff")][0].split('"')[1]
    panels = "/input/panels/" if os.path.isdir(u + "/panels") else "/input/"
    r = subprocess.run(["docker", "run", "--rm", "--platform", "linux/amd64", "--read-only", "--user", "65534:65534",
        "--cap-drop=ALL", "--security-opt", "no-new-privileges", "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=64m",
        "--pids-limit", "256", "--ulimit", "nofile=1024:1024", "--network=none", "-v", f"{u}:/input:ro", "-v", f"{out}:/output",
        "t2-forecaster:dev", "forecast", "--panels", panels, "--text", "/input/text/", "--asof", asof, "--out", "/output/forecast.parquet"],
        capture_output=True, text=True)
    name = os.path.basename(u)
    if r.returncode != 0: bad.append((name, "RUN", r.stderr[-300:])); continue
    if "[FALLBACK]" in r.stdout: fb.append(name)
    perm = subprocess.run(["find", out, "-mindepth", "1", "!", "-perm", "-o=r", "-o", "-type", "d", "!", "-perm", "-o=x"], capture_output=True, text=True).stdout.strip()
    if perm: bad.append((name, "PERM", perm))
    s = subprocess.run([sys.executable, SC, "score", "--card", u + "/card.toml", "--forecast", out + "/forecast.parquet"], capture_output=True, text=True)
    try:
        j = json.loads(s.stdout)
        if not j.get("admissible"): bad.append((name, "GATE", json.dumps(j.get("gates"))[:200]))
    except Exception: bad.append((name, "SCORER", (s.stdout + s.stderr)[-300:]))
    shutil.rmtree(out, ignore_errors=True)
print("units", len(units), "failures", len(bad), "fallbacks", len(fb))
for b in bad: print(b)
print("fallback units:", fb)
