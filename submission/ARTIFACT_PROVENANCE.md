# Artifact provenance (Agenthon 2026, Track 2)

## Executive summary
This image contains statistical code only. There are no fitted model files, no neural weights, no stored answers,
no lookup tables and no external data. The few numeric constants below were chosen by walk-forward backtests on
public panel history (rates / FX / factor panels from the public practice units, all dated 2000-2024), i.e. data that
existed before any sealed-set as-of date. The text corpus is not read and the House model is not called
(`models: []`).

## Components
| item | source / version | licence | role |
|---|---|---|---|
| t2agent (this repo) | our own code | MIT | forecast engine and CLI |
| qfbench2_track_forecasting (vendored, unmodified) | github.com/Agenthon-2026/track2-forecasting-public | MIT | panel and horizon helpers |
| qfbench2-common v2.4.4 | github.com/Agenthon-2026/Agenthon2026-public (tag v2.4.4) | MIT | contracts and limits |
| numpy 2.1.3, pandas 2.2.3, pyarrow 18.1.0, jsonschema 4.23.0 | PyPI | BSD-3 / Apache-2.0 / MIT | numerics |

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
- fallback ladder: if the engine fails, a plain Gaussian walk from the supplied history is written; output files are
  mode 0644 / directories 0755.

Backtest evidence (internal, public history only): the single-cell / multi-cell parameter split, the volatility
adjustment exponent and the t degrees of freedom were selected on as-of dates before 2015 and checked on 2015-2024.
Alternatives that were tested and rejected: GARCH(1,1), AR(1) centre, other drift look-backs, volatility
term-structure, robust volatility, correlation shrinkage, and any text-derived adjustment (no reliable signal).

## Data used for selection and calibration
Public panels up to 2024-12-18 (rates), 2024-10-31 (FX), 2024-05-31 (factors, macro). Pseudo-cards were generated from
those series at regular as-of dates; pseudo-cards coinciding (+-7 days, same asset) with a practice unit were excluded
from width-calibration experiments. No practice unit's hidden outcome was reconstructed or used.
