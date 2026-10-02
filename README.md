# Agenthon 2026 – Track 2 (Forecasting) · team 304

NeurIPS 2026 competition track "Verifiable AI for quantitative finance". Track 2 = probabilistic forecasting of
financial time series (rates, G10 FX, factors, monthly macro) from a numeric panel + a dated text corpus.
Score per card = composite(CRPS, variogram, tail pinball) / same composite of the text-blind baseline **M0**
(1.0 = M0, lower is better). Details: <https://www.agenthon.net/guides/>.

## Where we are (2026-10-02)
Dev-board scores (higher = better, M0 ≈ -1.0):

| image | what it is | dev score |
|---|---|---|
| v2 | baseline engine | **-0.9933** |
| v3 | v2 + transfer-card fix | **-0.9953** ← final candidate (v7 = v3 numerics + safer fallback path, outputs bit-identical) |
| v6 | single-cell cards ×0.9 narrower | -1.0094 |
| v4 | all cards ×1.35 wider | -1.0465 |

Width ≈ 1.0 is the dev optimum (wider and narrower are both worse). The dev leaderboard is a practice board; the
final ranking uses a sealed later window (one submission per track, 13–25 Oct 2026). Top dev scores (~-0.63) are
very likely driven by leakage between practice cards, not by honest skill – do not chase them.

## What the model is (no ML weights, no LLM calls; reads BLS release *numbers* from the corpus)
`submission/t2agent/engine.py`: random walk around the last observation + shrunk sample drift (trailing 300 steps),
volatility from the same window (with a mild recent-vol adjustment on single-cell cards), multivariate Student-t
shocks with a shared mixing variable (joint paths across assets/horizons), different parameter sets for
single-cell vs multi-cell cards (the variogram punishes departures from M0's joint structure), monthly-macro and
"transfer-card" special cases, and a never-crash fallback ladder (a failed card costs 4.0).
Full list of constants and what was rejected: `submission/ARTIFACT_PROVENANCE.md`.

## What we tested and rejected (all with walk-forward pseudo-cards + paired bootstrap)
GARCH(1,1), AR(1) centre, other drift look-backs, vol term-structure, robust vol, correlation shrinkage/windows,
per-panel tuning (overfits), fat-tail mixtures, uniform widening/narrowing (see table above).
**Text:** lexicon event study on FOMC statements/minutes/Beige Book/speeches, blind LLM labels (stress, pending
event, novelty, policy tone) and a direction test → no reliable incremental signal. Caveat: LLM "blinding" fails
(99 % of blind labellers guessed the year within 1 year), so those tests are optimistic upper bounds.
Lesson learnt: sanity-check effect sizes (a level-vs-change bug produced two fake "significant" results).

## Repo layout
- `submission/` – the image: `t2agent/` (engine + CLI), `Dockerfile`, `make_descriptor.py`, `run_all_units.py`
  (all 104 practice units through the official gates), `run_all_docker.py` (same, inside the image, with the
  platform's restrictions), `compare_ref.py`.
- `t2work/` – research code: `core.py` (M0 + scoring glue), `exp.py` (parallel search harness), `compare*.py`,
  `search*.py`, `selection_test*.py`, `transfer_test.py`, `text_study*.py`, `direction_test.py`, …
  (scripts use absolute paths under `/Users/apple/Documents/agenthon`; adjust `REPO`/paths in `t2work/data.py`.)

## Reproduce
```bash
git clone https://github.com/Agenthon-2026/Agenthon2026-public      # tag v2.4.4 for the toolkit
git clone https://github.com/Agenthon-2026/track2-forecasting-public
python3.13 -m venv .venv && . .venv/bin/activate
pip install "qfbench2-common[data] @ git+https://github.com/Agenthon-2026/Agenthon2026-public.git@v2.4.4#subdirectory=common"
pip install -e track2-forecasting-public
cd submission && PYTHONPATH=$PWD python run_all_units.py            # expect: failures 0, fallbacks 0
docker build --platform linux/amd64 -t t2-forecaster:dev .
```
Build **linux/amd64** (Apple-silicon builds are arm64 and will be held by the platform).

## Rules we keep (read the official Rules §6–§9)
- No lookup of any practice unit's hidden outcome; we calibrate on pseudo-cards made from public history only,
  excluding (asset, as-of ±7 d) pairs that coincide with practice units.
- Never commit the **Team Key**, tokens or passwords. The third-party corpora are not redistributable – clone the
  official repos instead of copying them here.
- Output files must be mode 0644 / dirs 0755 or the whole unit scores as `no_output`.
- Dev uploads: 5/day, 20 total; held uploads still count. Final phase: ONE submission.

## Team notes (2026-10-02)
- `research/nlp/` (Yakou): exact M0 replica + pseudo-card harness with common random numbers, oracle value-of-information
  table, blind-read tests, Fed-tone study; see `research/nlp/RESEARCH_LOG.md`. It reaches the same conclusion as
  `t2work/`: a text-blind statistical engine is worth ~0-1 % over M0 and the text "edge" on practice cards cannot be
  separated from model memory. NB: `research/nlp/t2agent/` is a research copy and is **not** the shipped package
  (`submission/t2agent/`); do not put both on `PYTHONPATH`.
- `submission/textlayer/` (Fed-tone centre shift for UST cells, House model first-token logprobs): merged for the record,
  **not wired into the image and not planned for the Final** (expected effect ~0; unverified `logprobs` support on the
  House route; no visibility into House calls on the platform). Re-open only with new evidence.
- Fallback path fixed in v7: a monthly macro card that fell back used the business-day key as a step count (~5x too wide).
  `submission/tests/test_fallback.py` guards it. Main-path outputs are unchanged (104/104 units bit-identical to v3).
- Policy to agree on: nothing that is packaged into the image, and no ship/no-ship decision, may depend on practice-unit
  outcomes reconstructed from sibling panels (Rules section 7 is ambiguous and the organisers have not answered issue #24).
  Use pseudo-cards from public history only.

## First real text edge (2026-10-02): numbers, not tone
`submission/t2agent/releases.py` reads the BLS release (CPI, core CPI, unemployment rate, payrolls) that the corpus
carries on/just before the as-of and the lagging monthly panel does not have yet, and appends it to the history (regexes,
no model). On public-history pseudo-cards one extra month is worth **~13 % of the score** (20 % at 2 steps) on such cards;
in the practice set 2/104 cards have a fresher release (CPI glidepath 2023, Sahm watch 2024) -> overall effect ~0.3 %.
Validation: `submission/tests/validate_releases.py` (54/60 BLS documents parse; core CPI 34/34, CPI 36/37, payrolls 11/11,
unemployment 11/12 within tolerance vs the public panel), unit tests in `submission/tests/`. Image `v8`.

## House-model fallback for number extraction (2026-10-02, image v9)
`t2agent/llm_extract.py` + `t2agent/house.py`: for monthly macro cards whose corpus has a release newer than the lagging panel
that the regexes cannot read (other country / publisher / layout), ONE narrow request asks the House model for the published
number plus the verbatim sentence; code verifies quote, number, month and plausibility before use; any failure = no change.
The model never forecasts. Stand-in evaluation (146 prompts): accepted answers 84/84 correct, 0/54 negative controls accepted,
4/4 synthetic foreign-format releases read. **Still to do: run `t2work/run_llm_extract_live.py` against the real Nemotron
(build.nvidia) and score it.** Descriptor must then declare the House model: `python make_descriptor.py --house ...`.
