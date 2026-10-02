# research/nlp — backtest harness and text-layer research

Research code behind `submission/textlayer/`. Every experiment, protocol (frozen before results) and number is in
`RESEARCH_LOG.md`. Nothing here is packaged into the image. The package named `t2agent` in this folder is
research-only and unrelated to `submission/t2agent`.

## Layout
- `t2agent/` — `task.py` (unit/pseudo-card representation), `m0.py` (exact replica of docs/M0-BASELINE.md, the
  score denominator), `engine.py` (text-blind engine used for ablations), `house.py`, `textlayer.py` (the failed v1
  LLM layer, kept for the record).
- `backtest/truth.py` — stitches all units' panels into full histories and resolves practice outcomes (offline).
- `backtest/evaluate.py` — scoring vs M0 (ratio per component, clip 4.0), pseudo-cards, cluster bootstrap CIs.
- `backtest/oracle*.py` — value of a directional/width signal of given accuracy under the ratio metric.
- `backtest/opus_*.py` — blind-read test (packets, scoring); `run_nemotron.py`, `eval_textlayer.py` — v1 layer.
- `backtest/fetch_fed.sh`, `fed_targets.py`, `fed_statements.py` — FOMC target changes and statements (offline).
- `backtest/nemotron_tone.py`, `step1_nlp.py`, `step1b_compare.py`, `c1_fx.py` — the Fed-tone study.
- `review/` — scoring the teammate engine (v2 copy, image sha256:2c9decfc…) in this harness.
- `tests/test_fault_injection.py` — v1 layer fault injection.

## Reproduce
Official repos cloned at the repo root (see top-level README), the shared venv, then from `research/nlp/`:
```bash
python backtest/truth.py                 # histories + practice outcomes -> backtest/data/ (gitignored)
bash backtest/fetch_fed.sh && python backtest/fed_targets.py && python backtest/fed_statements.py
python backtest/nemotron_tone.py         # needs .env.nvidia at the repo root: NVIDIA_API_KEY=... (gitignored)
python backtest/step1_nlp.py && python backtest/step1b_compare.py && python backtest/c1_fx.py
```
`backtest/data/`, `opus_packets/`, `opus_out/` are gitignored: they hold reconstructed practice outcomes, third-party
corpus excerpts and model outputs. Practice outcomes are used for local evaluation only (allowed: track issue #24);
nothing derived from them goes into the image.

## Headline findings
- The metric is anchored on M0: text-blind engines land at 0.98–1.01; perfect direction knowledge with a 0.5 sd
  shift reaches only ~0.81–0.86; a 0.1 sd shift breaks even at ~57–60% accuracy.
- LLM "edge" on practice cards is memory (Opus recognised ~all episodes even when masked; 82% hit rate when
  contradicting the trend vs 52.5% for the trend itself).
- Nemotron reads Fed tone far better than a lexicon (corr with the next Fed move .40/.60 vs .26), and adds
  out-of-time AUC on UST cards; monetised on top of the v2 engine it is neutral on random dates and helpful on
  authored event cards (thin n) — hence the capped `submission/textlayer`.
