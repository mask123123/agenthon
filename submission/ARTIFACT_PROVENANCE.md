# Artifact provenance (Agenthon 2026, Track 2)

## Executive summary
This image contains statistical code plus a small text layer. It holds no neural weights, no stored answers, no
lookup tables and no external data; its only fitted file is `textlayer/coefs.json` (two small logistic models). The few numeric constants below were chosen by walk-forward backtests on
public panel history (rates / FX / factor panels from the public practice units, all dated 2000-2024), i.e. data that
existed before any sealed-set as-of date. For UST targets the text layer reads the FOMC statements in the unit's own
`/input/text` and calls the House model (`models[]` discloses it and the fitted logistic).

## Components
| item | source / version | licence | role |
|---|---|---|---|
| t2agent (this repo) | our own code | MIT | forecast engine and CLI |
| qfbench2_track_forecasting (vendored, unmodified) | github.com/Agenthon-2026/track2-forecasting-public | MIT | panel and horizon helpers |
| qfbench2-common v2.4.4 | github.com/Agenthon-2026/Agenthon2026-public (tag v2.4.4) | MIT | contracts and limits |
| numpy 2.1.3, pandas 2.2.3, pyarrow 18.1.0, jsonschema 4.23.0 | PyPI | BSD-3 / Apache-2.0 / MIT | numerics |
| textlayer (this repo) | our own code; phrase lexicons written by the team | MIT | Fed-tone centre shift on UST cells |
| textlayer/coefs.json | fitted by us (research/nlp/backtest/step1_nlp.py, step1b_compare.py) | MIT | two L2 logistic models |
| nvidia/nemotron-3-super-120b-a12b rl-030326-fp8 | organizer House route (API), unchanged | per organizer | tone of FOMC statements |

## Method (engine.py) and constants
Random walk around the last observation (zero for log-return targets) with:
- drift = DRIFT x sample mean step over the trailing 300 steps; DRIFT = 0.5 for single-cell cards, 0.9 for multi-cell
  cards, 1.0 for monthly macro panels;
- step volatility from the same window, scaled by sqrt(steps); single-cell daily cards additionally scale it by
  (sd of last 21 steps / sd of window) ** 0.5, clipped to [0.5, 2];
- multivariate Student-t shocks (dof 8; 5 for monthly cards; 4 for transfer-style targets) with one shared mixing
  variable, so assets and horizons move together (covariance = sd_a sd_b corr_ab min(s_i, s_j));
- width multipliers: 1.05 on multi-cell cards; card family T2-F4 x1.10, T2-F2 x1.05 (a-priori, the card family is an
  allowed input); no extra transfer-card widening;
- transfer-style targets (history = early window + hole + as-of row): step volatility re-estimated from log returns
  (geometric mean of the target's early-window log-vol and the median log-vol of the other daily series in the panels
  at the as-of), early-window drift ignored;
- horizons converted to panel steps with the M0 rule (declared horizon kept unless the calendar-spacing count differs
  by 2x or more);
- fallback ladder: if the engine fails, an exact replica of the organizers' M0 baseline (docs/M0-BASELINE.md) is
  written (~1.0 by construction); if that cannot run either, a plain Gaussian walk (monthly cards in month steps);
  output files are mode 0644 / directories 0755.

## Text layer (textlayer/)
For UST targets only, when the engine succeeded: the latest FOMC statement (and the previous one) dated on or before
the as-of are taken from the unit's corpus. The House model gives two one-token tone readings (first-token logprobs,
thinking off, temperature 0; at most 6 requests per unit, normally 2). Features [sign of M0's drift, tone level, tone
change] -> frozen logistic model -> P(outcome above M0's centre) -> centre shift clip(0.25(2P-1), +-0.10) sd of the
cell's draws. Width and joint structure are unchanged; non-UST cells are untouched. If the House model does not
answer, the same features computed with a frozen phrase lexicon feed a second logistic model. Sentiment readings
(economic conditions, uncertainty, wording change) are written to the rationale only and do not move the forecast.

Backtest evidence (internal, public history only): the single-cell / multi-cell parameter split, the volatility
adjustment exponent and the t degrees of freedom were selected on as-of dates before 2015 and checked on 2015-2024.
Alternatives that were tested and rejected: GARCH(1,1), AR(1) centre, other drift look-backs, volatility
term-structure, robust volatility, correlation shrinkage; a free-form LLM forecasting layer (harmful), sentiment
features in the model (overfit), and a Fed-tone layer for USD FX (not robust).

## Data used for selection and calibration
Public panels up to 2024-12-18 (rates), 2024-10-31 (FX), 2024-05-31 (factors, macro). Pseudo-cards were generated from
those series at regular as-of dates; pseudo-cards coinciding (+-7 days, same asset) with a practice unit were excluded
from width-calibration experiments. Text-layer coefficients: fitted on 1,650 pseudo rates cards 2002-2024 with the same
exclusion, labels from public UST history, FOMC statements 2000-2024 from federalreserve.gov (public domain, offline
only, not in the image). Practice-unit outcomes were reconstructed from public panels for local evaluation of the
layer only (track issue #24); nothing derived from them is in the image. Model selection followed pre-registered
protocols recorded in research/nlp/RESEARCH_LOG.md.
