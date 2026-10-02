# Artifact provenance (Agenthon 2026, Track 2)

## Executive summary
This image contains statistical code only. There are no fitted model files, no neural weights, no stored answers,
no lookup tables and no external data. The few numeric constants below were chosen by walk-forward backtests on
public panel history (rates / FX / factor panels from the public practice units, all dated 2000-2024), i.e. data that
existed before any sealed-set as-of date. No language-model weights, adapters or NLP models are packaged. The unit's own text corpus is read for ONE purpose only:
extracting published numbers from macro releases (see "Text-derived numbers" below). That is done with regular expressions
and, only where they cannot read a release newer than the lagging panel, with ONE narrow request to the organiser-hosted House
model whose answer is verified in code before use. The descriptor therefore declares the House model (`models[]`, access api).

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
- fallback ladder: if the engine fails, a plain Gaussian walk from the supplied history is written, with horizons converted
  to panel steps like the main engine (explicit monthly steps, otherwise the M0 spacing rule); output files are
  mode 0644 / directories 0755.

Backtest evidence (internal, public history only): the single-cell / multi-cell parameter split, the volatility
adjustment exponent and the t degrees of freedom were selected on as-of dates before 2015 and checked on 2015-2024.
Alternatives that were tested and rejected: GARCH(1,1), AR(1) centre, other drift look-backs, volatility
term-structure, robust volatility, correlation shrinkage, and any text-derived adjustment (no reliable signal).

## Text-derived numbers (releases.py) - deterministic, no model
Monthly macro panels (CPI, core CPI, unemployment rate, nonfarm payrolls) lag the as-of by 1-2 months. If the unit's own
corpus contains a BLS release (`doc_type == "macro_release"`, dated on or before the as-of) for the month right after the
panel's last observation, the release's numbers are appended to the history: unemployment rate (level), CPI and core CPI
(seasonally adjusted monthly % change applied to the panel's last seasonally adjusted level), payrolls (change plus the
stated revisions of the two prior months). The engine then anchors on the newest value with one fewer step. A value is used
only if the release month is exactly the next month, it parses cleanly and passes domain bounds; otherwise nothing changes.
No judgement, tone or forecast is read from any text.

House-model fallback (llm_extract.py, house.py): used only for monthly macro cards, only when a macro_release document is dated
after the panel's last observation month and the regular expressions found nothing. One request per such asset (hard cap 4 per
unit, 120 s deadline, circuit breaker; the unit allowance is 25), temperature 0, seed 0, thinking disabled, <=300 output tokens.
The model must return a number plus the verbatim sentence it came from; code then checks that the sentence occurs in the text we
sent, that the number occurs in it (sign consistent with up/down wording; "unchanged" = 0), that the month is exactly the
month after the panel's last observation and is named in the text, and plausibility (monthly % change within 10 %; a stated
level only for rate-like series, never unadjusted). Any failure, error or missing endpoint leaves the forecast exactly as
without the call. The model never forecasts and no judgement of the model is used. Tests: tests/test_llm_extract.py (mocked),
tests/test_cli_house_e2e.py (real HTTP against a local mock of the route, incl. fabricated / wrong / garbage / 500 answers);
stand-in evaluation (t2work/eval_llm_extract.py): 146 prompts - BLS positives 84/84 accepted answers correct, 0/54 negative
controls accepted, 4/4 synthetic foreign-format releases read correctly, 2/2 absent cases refused. Parser validation (tests/validate_releases.py): 54 of the 60 unique
BLS documents in the public practice corpora parse; against the public monthly panel, core CPI 34/34, CPI 36/37, payrolls
11/11 and unemployment 11/12 are within tolerance (differences are later data revisions). Backtest on public history: one
extra month of macro data improves the score on such cards by ~13 % (20 % at two steps).

## Data used for selection and calibration
Public panels up to 2024-12-18 (rates), 2024-10-31 (FX), 2024-05-31 (factors, macro). Pseudo-cards were generated from
those series at regular as-of dates; pseudo-cards coinciding (+-7 days, same asset) with a practice unit were excluded
from width-calibration experiments. No practice unit's hidden outcome was reconstructed or used.
