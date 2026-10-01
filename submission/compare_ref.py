import glob, os, subprocess, sys, tempfile, json
import numpy as np, pandas as pd
units = sorted(glob.glob("/Users/apple/Documents/agenthon/track2-forecasting-public/units/t2-*"))
rows = []
for u in units:
    asof = [l for l in open(u + "/card.toml") if l.startswith("data_cutoff")][0].split('"')[1]
    panels = (u + "/panels" if os.path.isdir(u + "/panels") else u) + "/"
    o1, o2 = tempfile.mkdtemp(), tempfile.mkdtemp()
    a = subprocess.run([sys.executable, "-m", "t2agent.cli", "--panels", panels, "--text", u + "/text/", "--asof", asof, "--out", o1 + "/f.parquet", "--card", u + "/card.toml"], capture_output=True, text=True)
    b = subprocess.run([sys.executable, "-m", "qfbench2_track_forecasting.cli", "--panels", panels, "--text", u + "/text/", "--asof", asof, "--out", o2 + "/f.parquet", "--card", u + "/card.toml"], capture_output=True, text=True)
    if b.returncode != 0: rows.append(dict(unit=os.path.basename(u), note="ref failed")); continue
    x, y = pd.read_parquet(o1 + "/f.parquet"), pd.read_parquet(o2 + "/f.parquet")
    def w(d): return d.groupby(["asset", "horizon"]).value.agg(lambda s: np.quantile(s, .95) - np.quantile(s, .05))
    def m(d): return d.groupby(["asset", "horizon"]).value.median()
    ww = (w(x) / w(y)); dm = (m(x) - m(y)) / w(y)
    rows.append(dict(unit=os.path.basename(u), width_ratio=ww.median(), width_min=ww.min(), width_max=ww.max(), centre_shift_in_widths=dm.abs().max()))
d = pd.DataFrame(rows)
print(d.describe().round(3))
print("\nwidest vs reference:\n", d.sort_values("width_ratio").tail(6).round(3).to_string())
print("\nnarrowest vs reference:\n", d.sort_values("width_ratio").head(6).round(3).to_string())
print("\nlargest centre shifts:\n", d.sort_values("centre_shift_in_widths").tail(6).round(3).to_string())
