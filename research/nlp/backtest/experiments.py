"""Run named engine configs over real + pseudo cards and print a comparison table.

usage: python backtest/experiments.py <group>   (groups defined in GROUPS below)
"""

from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import replace

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from backtest.evaluate import load_all, run, summarize, cluster_ci  # noqa: E402
from t2agent.engine import Config  # noqa: E402

M0LIKE = Config(n_draws=500, w_recent=0.0, long_window=300, drift_shrink=1.0, lr_drift_shrink=0.0,
                corr_window=300, corr_shrink=0.0, seed=12345)
BASE = Config()

C1 = replace(BASE, w_recent=0.0, long_window=300, drift_shrink=1.0)  # M0 centre+vol, 4000 draws, our paths

CRN = replace(C1, crn=True, lr_drift_m0=True)

GROUPS = {
    "crn": {
        "crn_exact_m0": CRN,
        "crn_w0.9": replace(CRN, width=0.9),
        "crn_w0.95": replace(CRN, width=0.95),
        "crn_w1.05": replace(CRN, width=1.05),
        "crn_w1.1": replace(CRN, width=1.1),
        "crn_tau0.15": replace(CRN, tau=0.15),
        "crn_tau0.3": replace(CRN, tau=0.3),
        "crn_vol_ewma.3": replace(CRN, crn_vol=True, w_recent=0.3, ewma_hl=30, vol_mr_hl=120, long_window=300),
        "crn_vol_ewma.6": replace(CRN, crn_vol=True, w_recent=0.6, ewma_hl=20, vol_mr_hl=60, long_window=300),
        "crn_drift.5": replace(CRN, drift_shrink=0.5),
        "crn_drift0": replace(CRN, drift_shrink=0.0),
    },
    "center": {
        "m0like_4000": replace(M0LIKE, n_draws=4000),
        "C1": C1,
        "C1_w0.9": replace(C1, width=0.9),
        "C1_w1.1": replace(C1, width=1.1),
        "C1_w1.2": replace(C1, width=1.2),
        "C1_tau0.2": replace(C1, tau=0.2),
        "C1_tau0.35": replace(C1, tau=0.35),
        "C1_ewma.3": replace(C1, w_recent=0.3, ewma_hl=30, vol_mr_hl=120),
    },
    "noise": {
        "m0like_500_otherseed": M0LIKE,
        "m0like_4000": replace(M0LIKE, n_draws=4000),
    },
    "drift": {
        "base_drift0": BASE,
        "base_drift0.5": replace(BASE, drift_shrink=0.5),
        "base_drift1": replace(BASE, drift_shrink=1.0),
    },
    "vol": {
        "lr300_only": replace(BASE, w_recent=0.0, long_window=300),
        "lr1260_only": replace(BASE, w_recent=0.0, long_window=1260),
        "ewma20_mr60": BASE,
        "ewma60_mr120": replace(BASE, ewma_hl=60, vol_mr_hl=120),
        "blend.5_ewma30": replace(BASE, w_recent=0.5, ewma_hl=30),
    },
}


def main(group: str) -> None:
    real, pseudo = load_all()
    rows = []
    for name, cfg in GROUPS[group].items():
        for lab, cs in (("real", real), ("pseudo", pseudo)):
            df = run(cs, cfg)
            s = summarize(df, lab)
            lo, hi = cluster_ci(df)
            s.update({"rm": df["r_marg"].mean(), "rj": df.loc[df.ncell > 1, "r_joint"].mean(), "rt": df["r_tail"].mean()})
            s.update({"cfg": name, "ci": f"[{lo:.3f},{hi:.3f}]"})
            rows.append(s)
            df.to_csv(pathlib.Path(__file__).parent / "data" / f"res_{group}_{name}_{lab}.csv", index=False)
    t = pd.DataFrame(rows).set_index(["cfg", "label"])
    cols = ["n", "mean", "rm", "rj", "rt", "ci", "median", "n_clip", "F1", "F2", "F3", "F4", "early(<2016)", "late(>=2016)"]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(t[cols])


if __name__ == "__main__":
    main(sys.argv[1])
