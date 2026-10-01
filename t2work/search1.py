import sys, json, copy
sys.path.insert(0, "/Users/apple/Documents/agenthon/t2work")
import exp
df = exp.init()
print({p: int((df.panel == p).sum()) for p in df.panel.unique()}, "shock share", df.shock.mean().round(3), flush=True)
base = {}
grids = {"drift": [0.25, 0.5, 0.75, 1.0], "k": [0.9, 1.0, 1.1, 1.2], "w_long": [0, 0.25, 0.5], "beta": [0, 0.25, 0.5],
         "hurst": [0.45, 0.5, 0.55], "nu": [4, 5, 8], "window": [200, 300, 450]}
best = {}
for panel in ["rates", "fx", "factors", "macro"]:
    P = {} if panel != "macro" else {}
    key_drift = "monthly_drift" if panel == "macro" else "drift"
    key_k = "monthly_k" if panel == "macro" else "k"
    cur = {}
    base_out = exp.evaluate({panel: cur}, df, panels=[panel]); b = exp.summarize(base_out)
    print(panel, "baseline", {k: round(v, 4) for k, v in b.items()}, flush=True)
    for sweep in range(2):
        for name, vals in grids.items():
            nm = {"drift": key_drift, "k": key_k}.get(name, name)
            if panel == "macro" and name in ("w_long", "beta", "window"): continue
            scored = []
            for v in vals:
                trial = {**cur, nm: v}
                s = exp.summarize(exp.evaluate({panel: trial}, df, panels=[panel]))
                scored.append((s["obj"], v, s))
            scored.sort(key=lambda t: t[0])
            cur[nm] = scored[0][1]
        s = exp.summarize(exp.evaluate({panel: cur}, df, panels=[panel]))
        print(panel, "sweep", sweep, cur, {k: round(v, 4) for k, v in s.items()}, flush=True)
    best[panel] = cur
json.dump(best, open("best1.json", "w"), indent=1)
print("BEST", json.dumps(best))
