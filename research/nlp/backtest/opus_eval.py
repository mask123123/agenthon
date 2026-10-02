"""Score the Opus text-edge test under the pre-registered protocol (RESEARCH_LOG.md, 2026-10-02).

Primary: on M0's exact draws shift each cell by k*sd_M0 in the stated direction,
k = 0 (dir 0 or p < .6), 0.1 (.6 <= p < .75), 0.25 (p >= .75). Hit rate vs sign(y - last value).
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.evaluate import cluster_ci, components, load_all, normalized  # noqa: E402
from t2agent import m0  # noqa: E402

OUT = ROOT / "backtest" / "opus_out"


def k_of(d: int, p: float) -> float:
    if d == 0 or p < 0.6:
        return 0.0
    return 0.1 if p < 0.75 else 0.25


def evaluate(arm: str, real: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pmap = json.loads((ROOT / "backtest" / "data" / "opus_map.json").read_text())
    inv = {u: p for p, u in pmap.items()}
    by_id = {c["task"].unit_id: c for c in real}
    cells_rows, card_rows = [], []
    for uid, c in by_id.items():
        f = OUT / arm / f"{inv[uid]}.json"
        if not f.is_file():
            continue
        t, y = c["task"], c["y"]
        got = json.loads(f.read_text())
        view = {(e["asset"], int(e["horizon"])): e for e in got.get("cells", [])}
        base, cells, info = m0.m0_parts(t)
        shift = np.zeros(len(cells))
        for i, (a, h) in enumerate(cells):
            e = view.get((a, h), {})
            d, p = int(e.get("direction", 0) or 0), float(e.get("confidence", 0.5) or 0.5)
            last = 0.0 if t.target_type == "log_return" else float(t.history(a).iloc[-1])
            truth = np.sign(y[i] - last)
            shift[i] = k_of(d, p) * info["sd"][i] * d
            cells_rows.append({"uid": uid, "family": t.category[-2:], "dir": d, "p": p, "hit": float(d == truth) if d else np.nan,
                               "basis": e.get("basis", ""), "vol_view": e.get("vol_view", "")})
        sc = normalized(components(base + shift[None, :], y), c["ref"], len(y))[0]
        card_rows.append({"uid": uid, "family": t.category[-2:], "asof": t.asof, "year": int(t.asof[:4]),
                          "score": sc, "recognized": str(got.get("recognized", "none")).lower() not in ("none", "", "no")})
    return pd.DataFrame(cells_rows), pd.DataFrame(card_rows)


def report(arm: str, cells: pd.DataFrame, cards: pd.DataFrame) -> None:
    d = cells[cells["dir"] != 0]
    print(f"\n=== arm {arm}: {len(cards)} cards, {len(cells)} cells, {len(d)} with a direction")
    print(f"hit rate (all directional cells): {d['hit'].mean():.3f}")
    for lo, hi in ((0.5, 0.6), (0.6, 0.75), (0.75, 1.01)):
        s = d[(d["p"] >= lo) & (d["p"] < hi)]
        print(f"  confidence [{lo},{hi}): n={len(s):3d} hit={s['hit'].mean():.3f}")
    for b in ("established", "inferred"):
        s = d[d["basis"] == b]
        print(f"  basis {b:11s}: n={len(s):3d} hit={s['hit'].mean():.3f}")
    lo, hi = cluster_ci(cards)
    print(f"score (pre-registered mapping): mean={cards['score'].mean():.3f} CI[{lo:.3f},{hi:.3f}] "
          f"median={cards['score'].median():.3f}  recognized={int(cards['recognized'].sum())}/{len(cards)}")
    print("  by family:", cards.groupby("family")["score"].mean().round(3).to_dict())
    print("  early(<2016) / late(>=2016):", round(cards[cards.year < 2016].score.mean(), 3), "/",
          round(cards[cards.year >= 2016].score.mean(), 3))


if __name__ == "__main__":
    real, _ = load_all()
    res = {}
    for arm in sys.argv[1:] or ["A", "B"]:
        cells, cards = evaluate(arm, real)
        res[arm] = (cells, cards)
        report(arm, cells, cards)
    if "A" in res and "B" in res:
        a, b = res["A"][1].set_index("uid")["score"], res["B"][1].set_index("uid")["score"]
        common = a.index.intersection(b.index)
        diff = (a[common] - b[common])
        print(f"\nA - B on {len(common)} common cards: {diff.mean():+.3f} (negative => named arm better => memory)")
