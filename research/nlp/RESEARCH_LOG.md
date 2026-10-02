# Track 2 research log

## 2026-10-02 — backtest harness + first engine ablations

**Harness.** `backtest/truth.py` stitches every unit's panels (exemplar excluded: synthetic).
Daily panels agree exactly across units; macro_monthly differs by vintage (expected).
Outcomes: target date = `np.busday_offset(asof, h)`; level = last value <= target (<= 5d stale);
log_return = sum(log1p(r)) over (asof, target]. 93/103 practice units fully resolvable
(10 not: EM transfer targets, 2024 cards whose targets pass the data end, one monthly).
`backtest/evaluate.py`: score = sum_k w_k * ours_k / M0_k, clip 4.0, single-cell weights (5/7, 0, 2/7);
M0 = exact replica of docs/M0-BASELINE.md (`t2agent/m0.py`). Pseudo-cards: each practice template
(assets, horizons, target type, panels) re-dated to 12 random as-ofs 2002-2024 -> 1128 cards.

**Noise floor.** M0's distribution with a different seed: real 0.980, pseudo 1.014 (500 draws);
4000 draws: 0.982 / 1.008. Cluster-bootstrap 95% CI half-width ~0.01-0.02.

**Independent sampling (fresh draws).** No configuration beats ~0.98 real / ~0.99 pseudo.
Removing M0's drift hurts (real 1.077 vs 1.006). Joint ratio is a lottery: M0's variogram score is
near zero on some cards, so even an identical distribution averages joint ratio ~3.9 on pseudo.

**Common random numbers (transform M0's exact draws).** Exact M0 = 1.000 by construction.
Width 0.95: real 0.996, pseudo 0.995. Scale-mixture tails, EWMA vol, drift shrink: all within
+-0.01 of 1.0 or worse. Conclusion: a text-blind statistical engine is worth ~0-1%.

**Oracle value of information (`backtest/oracle.py`, uses outcomes, analysis only).** Shift centre
by k*sd in a direction correct with prob p (per card):

| p | k=0.1 | k=0.25 | k=0.5 |
|---|---|---|---|
| 0.5 | 1.01 | 1.04-1.07 | 1.16-1.22 |
| 0.6 | 1.00 | 1.01-1.03 | 1.10-1.15 |
| 0.7 | 0.98 | 0.97-0.99 | 1.01-1.08 |
| 1.0 | 0.94 | 0.88 | 0.81-0.86 |

"Big move" width signals lose unless ~90%+ accurate. Development-board scores near 0.63 are below
what perfect direction knowledge gives, i.e. they require knowing magnitudes (answer leakage or
memory) and should not transfer to the sealed Final.

## 2026-10-02 — Opus text-edge test: pre-registered protocol (frozen before any output is read)

Packets: `backtest/opus_packets.py` (93 resolvable cards, anonymous ids P001.., docs renamed, author prose
withheld, 14k chars/card). Arm A named; arm B masked (dates/years/months/officials/event words, relative doc
dates, no levels). Masking is partial (company names, tickers and context can still identify episodes).

Reader output per (asset, horizon): direction in {-1,0,+1} vs the last observed value (log_return: sign of the
cumulative return), confidence p in [0.5, 1], vol_view in {lower, normal, higher}, basis in {established, inferred}.

Scoring (fixed now): on M0's exact draws, shift the cell centre by k*sd_M0 in the stated direction, with
k = 0 if direction == 0 or p < 0.6; k = 0.1 if 0.6 <= p < 0.75; k = 0.25 if p >= 0.75. Width untouched.
Report: directional hit rate (all, and by confidence bucket) vs the realized change from the last value,
normalized score vs 1.0 with cluster-bootstrap CI, by family, and arm A minus arm B (memory indicator).
Secondary (exploratory, not used for decisions): vol_view -> width x0.9 / x1.0 / x1.1.

## 2026-10-02 — Opus text-edge test: results

8 Opus readers (2 arms x 4 batches), 93 cards, 240 cells. Masking failed: episodes recognized 91/93 (A) and
93/93 (B) — BLS release ids, FOMC voter lists, 8-K filings, unmasked spaced digits. A-B = -0.001 (uninformative).

| | A named | B masked |
|---|---|---|
| hit rate, all directional cells | 0.708 | 0.680 |
| hit, confidence 0.6-0.75 | 0.955 (n=44) | 0.900 (n=50) |
| hit, basis established | 0.919 (n=37) | 0.852 (n=27) |
| score, pre-registered mapping | 0.981 [0.973, 0.988] | 0.982 [0.973, 0.990] |
| score, exploratory k=.25/.5 | 0.959 [0.940, 0.976] | 0.962 [0.943, 0.980] |

Control: text-free trend sign (M0 drift) hits 0.525. Opus contradicting the trend hits 0.818 (A) / 0.754 (B):
correct reversals at that rate are a hindsight signature. Conclusion: on practice cards the text "edge" is not
separable from memory; even with memory the gain is 2-4%. On the sealed Final (no memory) expect ~0.

## 2026-10-02 — review of teammate engine v2 (ghcr.io/mask123123/t2-forecaster@sha256:2c9decfc...)

Harness: real 0.987 [0.959, 1.014], pseudo 0.994 [0.981, 1.009] (matches the reported dev 0.99). Best engine so far.
Seed = crc32(unit_id) = M0's seed, so on single-cell cards its first 500 normals equal M0's (common random numbers).
Ablations: Student-t(8) helps (Gaussian: 1.010/1.016); beta=.5 helps (beta 0: .996/1.005); single drift .5 beats 1.0
on pseudo (1.012); family k helps real (F4 .971 vs 1.007) but hurts pseudo (.978 without) — event-selection prior,
the one overfit-risk knob. Multi-cell k=1.0 instead of 1.05: real .954, pseudo 1.002 (vs .972 / 1.006), within CI.
Hazards: fallback on monthly card cpi-glidepath -> 4.00 (uses BD horizon key as steps); fallback with an asset
missing from hist anchors at 0 -> 4.00. ARTIFACT_PROVENANCE says NU=5 and omits F2 k=1.05 (code: nu=8, F2 1.05).

## 2026-10-02 — Nemotron structural-fact layer v1: schema and mapping (frozen before any run)

API facts measured: build.nvidia `nvidia/nemotron-3-super-120b-a12b` works with `chat_template_kwargs.enable_thinking`;
reasoning tokens count inside completion_tokens (so the 4,000 cap is shared by thinking + answer). Corpora are large
(per card p50 308k chars, max 490k; per doc p90 122k) -> map-reduce.

Pipeline: map = one call per document (newest first, <= 10 docs, each cut to head 24k + tail 6k chars), thinking off,
max_tokens 1200 -> doc facts JSON. Reduce = one call with targets, numeric summary and all doc facts, thinking on,
max_tokens 4000 -> card JSON. Retry policy: one retry per failed call, total cap 14 requests, unit deadline 900 s.

Card schema v1 (per target asset):
  established_action: {direction -1|0|+1, confidence 0.5-1, what, evidence}  # decided/announced facts only
  binary_event_in_horizon: {present bool, what, evidence}                    # scheduled decision/vote/election
  stress: calm|elevated|crisis
  regime_constraint: {type none|peg|floor|ceiling|band|cap, level, evidence}  # recorded only in v1, not mapped

Mapping (applied in code to any engine's draws, per cell; sd = std of the engine draws for that cell):
  shift  = 0.10 * sd * direction  if established_action.confidence >= 0.70 else 0
  width  = 1 + 0.08*[binary_event_in_horizon] + 0.08*[stress == crisis], capped at 1.15, scaling about the median
  any failure (no endpoint, budget, timeout, unparseable reduce) -> no overlay (pure engine draws)
Success criteria for this run (25 cards, stratified): reduce JSON parse rate >= 95% after retry; p95 wall time per
card < 600 s at 4 parallel calls; report signal frequencies; score effect on the teammate engine is reported but
with n=25 it is expected to be inside noise (decision-relevant only if it is clearly negative).

## 2026-10-02 — Nemotron structural-fact layer v1: results (FAILED pre-registered criteria)

25 cards (stratified), build.nvidia stand-in. Reliability: 21/25 extracted (5 reduce calls hit the 4,000-token cap,
thinking consumed it); 0 transport failures; seconds p50 45 / p95 125 / max 244; requests p50 8 / max 12.
Signals: established action on 26/32 asset entries, 25 of them self-rated >= 0.7 (uncalibrated: hit rate 0.585 on
41 cells, with the model's own memory of these episodes); binary event flagged on 20/32; regime constraint 0/32.
Score on teammate engine: base 0.9881 -> v1 overlay 1.0224 (+0.034). Shift only 1.0074, width only 1.0038:
both components harmful. Decision: no text overlay in the submission. Fault-injection suite: 12/12 pass.

## 2026-10-02 — Step 1: traditional-NLP text test (protocol frozen before any result)

Data (offline only, never packaged): federalreserve.gov open-market tables -> 110 dated target changes 1990-2026
(`backtest/fed_targets.py`); 213 FOMC statements 2000-2024 (`backtest/fed_statements.py`). FRED timed out from CLI.

Universe: rates cards (all target assets UST_*). Pseudo-cards: each rates-only practice template re-dated to 40 random
as-ofs 2002-2024 (seed 23). Real: resolvable rates practice cards.
Features at as-of t (information <= t only): s1 = sign of last target change; s2 = sum of changes in prior 180 days
(pct points); s3 = min(days since last change / 365, 2); x1 = (hawk - dove) / (hawk + dove + 1) phrase counts in the
latest statement <= t; x2 = x1(latest) - x1(previous). Lexicon v1 written from domain knowledge, frozen, not tuned.
Label per cell: z = sign(y - M0 mean). Model: L2 logistic regression (C = 0.3), features standardized, per-cell rows.
Mapping: shift = clip(0.25 * (2P - 1), -0.10, 0.10) * sd_M0, applied to M0's exact draws (so baseline = 1.000).
Validation: fit as-of < 2013-01-01, test >= 2013; and the reverse. Models: 1a state (s1-s3), 1b state + lexicon,
1c lexicon only, control = trend sign of M0 drift only. Report test accuracy, AUC, score delta with cluster CI.
Kill criterion: if 1b's test delta CI includes 0 in both splits, text features from statements do not enter v2.

## 2026-10-02 — Step 1 results

1720 pseudo rates cards + 41 real. Test accuracy of the side of y vs M0 centre: 0.58-0.69, AUC 0.54-0.80.
Score deltas (shift = clip(0.25(2P-1), +-0.1) sd on M0 draws), pseudo: 1a +0.005 / -0.001, 1b +0.005 / -0.003,
1c +0.005 / -0.002 (all CIs include 0, both splits) -> kill criterion met on pseudo; real (n=29 / 12): 1b +0.003 /
-0.035 [-0.067,-0.012]. Lexicon weights are negative on hawkish tone: largely a proxy for trend reversal.
Exploratory (not pre-registered): M0's drift sign is on the WRONG side of the outcome 61-67% of the time (M0
over-extrapolates its 300-day mean step). Fixed fade rule (shift -0.1 sd x sign(M0 drift)): pseudo 0.9957, real
0.9822. Incremental AUC of lexicon over the trend feature: 0.601 -> 0.696 (train<2013), 0.666 -> 0.680 (train>=2013).
Reading: text carries some information beyond the numbers, but monetized under the ratio metric it is worth <~1%.

## 2026-10-02 — Step 1b: Nemotron tone vs lexicon, plus sentiment features (protocol frozen before results)

Nemotron (build.nvidia stand-in), thinking off, temperature 0, max_tokens 1, top_logprobs 10, per statement:
  n1 = expected value over digit tokens 1..5 of "how hawkish is this statement" (1 very dovish .. 5 very hawkish)
  n2 = P(more hawkish) - P(more dovish) from tokens A/B/C when shown the previous and the current statement
New traditional features (lexicons written from domain knowledge, frozen): x3 = economic-condition sentiment
(pos - neg)/(pos + neg + 1); x4 = uncertainty phrases per 1,000 words; x5 = 1 - Jaccard(word trigrams, previous).
Same data, splits, logistic model (C = 0.3) and mapping as Step 1. Every model includes the trend feature.
Models: T (trend), T+L (x1,x2), T+L+S (x1..x5), T+N (n1,n2), T+N+L+S (all). Report test AUC, accuracy, delta + CI.
Sanity check (tone validity): correlation of n1 and x1 with the next 90-day Fed target change.
Decision rule: N earns a place in the runtime layer only if T+N beats T+L+S on test AUC in both splits.

## 2026-10-02 — Step 1b results

Nemotron tone scored 213/213 statements (n2 211/211; logprob mass on the answer tokens ~0.999).
Tone validity vs next-90-day Fed target change: lexicon x1 +0.26, Nemotron n1 +0.40, Nemotron n2 (vs previous) +0.60.
Side-of-M0 prediction, test AUC (train<2013 / train>=2013): T .601/.666, T+L .696/.680, T+L+S .610/.580,
T+N .680/.708, all .636/.625. Decision rule met: T+N > T+L+S in both splits. Sentiment features (x3-x5) overfit: dropped.
On M0 base, T+N pseudo delta -0.004 / -0.009 (CIs include 0). On the TEAMMATE engine base (what we would ship):
pseudo T+N +0.003 / +0.004, N only +0.001 / +0.002, L only -0.004 / +0.003 (all CIs include 0, n~850 each);
real rates cards: T+N -0.002 (n=29) / -0.027 [-0.056,-0.003] (n=12), N only +0.004 / -0.034 [-0.058,-0.012].
Reading: Nemotron reads Fed tone clearly better than the lexicon, but on random dates it adds nothing beyond the
teammate engine; on authored event cards there is a hint of value (thin n). Coverage: every practice unit ships >= 1
FOMC statement; 29/43 rates units ship >= 2 (needed for n2).

## 2026-10-02 — Step C1: Fed tone -> USD FX (protocol frozen before results)

Cards: FX-only practice templates (all targets in g10_fx_daily) re-dated to 40 random as-ofs 2002-2024 (seed 29);
real: resolvable FX-only practice cards. Orientation o = -1 for USD-per-unit quotes (AUD, EUR, GBP, NZD), +1 for
units-per-USD (CAD, CHF, DKK, JPY, NOK, SEK) — H.10 conventions, checked on levels.
Features per cell: trend (sign of M0 drift), o*n1c, o*n2 where n1c = n1 - 3 (Nemotron tone, latest statement <= as-of).
Same logistic (C = 0.3), mapping (shift = clip(0.25(2P-1), +-0.1) sd), splits, base = teammate engine.
Ship rule (same standard as the rates layer A): pseudo delta CI upper <= +0.010 in both splits AND pooled real delta < 0.
Final coefficients (rates and, if shipped, FX) are refit on ALL pseudo-cards excluding (asset, as-of +-7 days) pairs that
coincide with a practice unit (team rule), then frozen into the image.

## 2026-10-03 — correction, lexicon fallback, release wiring

Correction: the C1 protocol said "same standard as the rates layer A: pseudo CI upper <= +0.010". Layer A itself has
pseudo CI uppers +0.0126 / +0.0138 (it was accepted on the user's decision with that risk stated, not on that rule).
C1's rejection stands on its real-card result (+0.015).
Lexicon fallback (trend + lexicon tone, teammate engine base): pseudo +0.0033 [-0.0060,+0.0140] / +0.0049
[-0.0039,+0.0151]; real -0.0014 (n=29) / -0.0117 (n=12), pooled -0.0044 — same risk profile as layer A, weaker on
real cards. Shipped only as the path used when the House model does not answer. Sentiment features stay out of the
model (Step 1b: they lowered AUC) and are reported in the rationale only.
Release (branch feature/fed-tone-textlayer): exact-M0 fallback (bit-identical to t2agent/m0.py on 5 card types),
Gaussian fallback in month steps for monthly cards, text layer wired on the engine path. End-to-end on the 103 practice
units, 93 resolvable scored vs M0: offline (lexicon path) 103/103 admissible, untouched cards identical, 0.9867 ->
0.9834 (lexicon cards 0.9873 -> 0.9805, 23 better / 22 worse); live stand-in on UST units 43/43 admissible, 0.9852 ->
0.9818 (23 House, 20 lexicon under stand-in rate limits). Practice cards informed the decisions: optimistic.

## 2026-10-03 — Ratio-aware forecasting (metric structure, no new information)

Idea: the card score is E[ours(y) / M0(y)] and M0's draws are reproducible at run time, so under belief p the optimal
forecast for the marginal CRPS term is q(y) ∝ p(y) / CRPS_M0(y) (CRPS is proper). Implemented as importance
resampling of the engine's joint draws (rows) with weights M0_loss(x)^-gamma (`backtest/ratio_aware.py`;
CRPS helper matches the official estimator to 1e-15). Belief = teammate engine v2.
- gamma 0 / .25 / .5 / 1.0, pseudo: 0.9945 / 0.9670 / 0.9572 [0.943, 0.973] / 0.9724; real: 0.9867 / 0.9856 / 1.0099 /
  1.0814. Widening before reweighting (k 1.15-1.5) is worse everywhere.
- The split is on single-cell cards: pseudo single-cell gains 5-7% at gamma .5 for every asset class (66-76% of cards
  better); practice single-cell cards (event-selected) lose. Multi-cell improves or holds on both sets.
- F1-only routing looked good on real (19 cards) but pseudo F1 templates get worse (+0.013): rejected as overfit.
- Belief choice: engine > 50/50 pool > M0's own draws.
- Robust pick (min over max(real, pseudo), single/multi gammas): 0.10 / 0.25 -> real 0.9809, pseudo 0.9792
  [0.967, 0.993] (788 better / 338 worse). Aggressive 0.5 / 0.25 -> real ~1.007, pseudo ~0.958 (a bet that Final cards
  look like random dates rather than event-selected practice cards). Selected in-sample (2 parameters).
