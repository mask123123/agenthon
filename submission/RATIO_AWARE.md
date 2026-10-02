# Ratio-aware resampling — what it is, why it works, how to check it

**One line:** after the engine draws, we resample those draws (whole rows) toward the region where M0 would score
well, because the leaderboard divides our loss by M0's loss and M0's draws are known at run time.
Code: `t2agent/ratio_aware.py` (≈90 lines), hook in `t2agent/cli.py`, tests in `tests/test_ratio_aware.py`.

## Why
Each card's score is `Σ_k w_k · ours_k(y) / M0_k(y)` (marginal CRPS, variogram, tail pinball), averaged over cards.
For a belief `p(y)` about the outcome, the expected marginal ratio is

    E_p[ CRPS_ours(y) / CRPS_M0(y) ]  =  ∫ p(y) / CRPS_M0(y) · CRPS_ours(y) dy

so it is the expected CRPS under the reweighted belief `q(y) ∝ p(y) / CRPS_M0(y)`. CRPS is proper, so the best
forecast for the ratio is **q, not p**: put a bit more mass where M0 happens to be good (its denominator is small
there, and that is where our ratio blows up). M0 is the organizers' published, reproducible baseline
(`docs/M0-BASELINE.md`; our `t2agent/m0_fallback.py` reproduces its draws exactly), so `CRPS_M0(y)` is a known
function of `y` for every card. This uses no extra information about the outcome — it only optimizes the published
scoring formula. (The organizers' M0 doc acknowledges that a reproducible baseline is "one more handle on a scored
board".)

## How
1. `m0_ref = m0_fallback.m0_draws(hist_panel, ...)` — M0's exact 500 draws, from the panels **as shipped** (not the
   history extended with BLS releases, because the organizers' M0 does not see those).
2. For every engine draw `x` (one row = all cells): `L(x) = mean_cells CRPS_M0,cell(x) / CRPS_M0,cell(M0 median)`.
3. Weights `w = L(x)^-γ`, systematic resampling of the rows (seeded by the unit id, deterministic). Rows are kept
   whole, so the joint structure of the engine is preserved; n_draws is unchanged.
4. `γ = 0.10` on single-cell cards, `γ = 0.25` on multi-cell cards. Engine path only (never on the M0 / Gaussian
   fallbacks); applied before the text layer. Any failure → draws unchanged; the rationale says what happened.

## Evidence (`research/nlp/RESEARCH_LOG.md`, 2026-10-03; teammate engine v2 as the belief)
| | γ = 0 (before) | γ single .10 / multi .25 (shipped) | γ .5 everywhere (aggressive) |
|---|---|---|---|
| 1,128 pseudo-cards (random dates, public history) | 0.9945 | **0.9792** [0.967, 0.993], 788 better / 338 worse | 0.9572 |
| 93 practice cards with resolvable outcomes | 0.9867 | **0.9809** | 1.0099 |

- The two sets disagree on single-cell cards: on random dates every asset class gains 5–7 % at γ .5; the practice
  single-cell cards (authored around shocks, outcomes in the tails) lose. γ was chosen to minimize the worse of the two
  scores, so the shipped setting improves both. If the Final cards turn out to look like random dates, a larger γ
  would have gained more; if they look like the practice set, it would have lost — the shipped γ is the hedge.
- Widening the engine before reweighting is worse everywhere; using M0's own draws as the belief is worse than the
  engine.
- Selected in-sample (2 parameters on these two sets).

## End-to-end checks on this branch (full CLI: engine + ratio-aware + text layer, offline -> lexicon path)
| check | result |
|---|---|
| 103 practice units, official gates | 103/103 admissible, 0 fallbacks |
| 93 resolvable practice units, score vs M0 | `main` (v9) 0.9831 -> **0.9813**; UST units 0.9767 -> 0.9716; 38 better / 55 worse |
| linux/amd64 image, all 104 units, `--network=none --read-only --user 65534 --cap-drop=ALL` | 104/104 admissible, 0 fallbacks |

On the practice cards the gain on top of v9 is small (part of it overlaps with the text layer, and these event-selected
cards are where the resampling helps least); the larger gain is on random-date pseudo-cards (0.9945 -> 0.9792), which no
text layer touches.

## If you change something
- Changing the engine changes the belief `p`; re-run `research/nlp/backtest/ratio_fine.py` (uses the engine in
  `research/nlp/review/`) before trusting γ.
- To switch it off: set `GAMMA_SINGLE = GAMMA_MULTI = 0.0` in `t2agent/ratio_aware.py` (the draws are then returned
  unchanged).
