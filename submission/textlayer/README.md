# textlayer — Fed-tone centre shift for UST cells

A small, capped text layer on top of the statistical engine. It reads the unit's own corpus, asks the House model
two one-token questions about the latest FOMC statement, and nudges the centre of **UST cells only** by at most
**0.10 sd**. Width, joint structure and every non-UST cell are left untouched. Any failure leaves the draws exactly
as the engine produced them.

## What it does, per unit
1. `find_statements`: FOMC statements in `text/corpus_index.json` (`doc_type == "fomc_statement"`, or file name
   fallback) dated on or before the as-of; takes the latest and the one before it (8,000 chars each).
2. `tone` (House, thinking off, `max_tokens=1`, first-token logprobs):
   - `n1` = expected digit for "how hawkish is this statement" (1 very dovish … 5 very hawkish);
   - `n2` = P(more hawkish) − P(more dovish) versus the previous statement (neutral if there is no previous one).
   If the route returns no logprobs, the answered token is used with probability 1.
3. For each UST asset: `x = [sign of M0's drift (trailing 300 steps), n1, n2]` → frozen logistic model
   (`coefs.json`) → `P(outcome above M0's centre)` → `shift = clip(0.25·(2P−1), ±0.10) × sd(cell draws)`.

Budget: 2 House requests per unit (3 with the one retry on the first call), hard cap 6, 240 s text-layer deadline,
circuit breaker after 3 consecutive failures. Measured with the build.nvidia stand-in: 1–3 s per unit.

## Evidence (research/nlp/RESEARCH_LOG.md has every number and the pre-registered protocols)
- Tone validity vs the next-90-day Fed target change: lexicon +0.26, Nemotron `n1` +0.40, `n2` +0.60.
- Side of the outcome vs M0's centre, out-of-time (train <2013 / ≥2013), AUC: trend only .60/.67,
  trend + Nemotron .68/.71, trend + lexicon + sentiment .61/.58 (sentiment features overfit and were dropped).
- On top of the v2 engine, 1,720 pseudo rates cards (random dates): +0.001 to +0.004, CIs include 0 (neutral).
  Real practice rates cards: −0.002 (n=29, as-of ≥2013) and −0.027 [−0.056, −0.003] (n=12, as-of <2013).
  Read: neutral on random dates, a hint of value on authored event cards; downside bounded by the 0.10 sd cap.
- Runtime check: the tone the layer reads from practice corpora matches the training readings of the same
  statements (n=10, corr 0.996, mean |diff| 0.06 on the 1–5 scale).
- Rejected: the same idea for USD FX (pseudo neutral, real +0.015 worse, unstable signs) — not shipped.

Coefficients (`coefs.json`): fitted on 1,650 pseudo rates cards 2002–2024 (practice-unit (asset, as-of ±7 d) pairs
excluded), labels from public UST history; Nemotron tone of 213 public-domain FOMC statements 2000–2024.
The prompts in `fed_tone.py` are byte-identical to the fitted ones; a test fails if they drift.

## Wiring it in (not done in this PR — owner of `t2agent/cli.py` and `Dockerfile` to apply)
`t2agent/cli.py`, right after the final `np.isfinite` safety net and before `_rationale(...)`:
```python
    text_ledger = []
    if not res.meta.get("fallback"):
        try:
            from textlayer import text_overlay
            samples, text_ledger = text_overlay(samples, assets, horizons, hist, a.text, a.asof)
        except Exception as exc:
            text_ledger = [f"text layer unavailable: {type(exc).__name__}"]
    rat = _rationale(unit_id, a.asof, assets, horizons, family, n_draws, samples, res, note)
    if text_ledger:
        rat += "\n## Text layer (Fed tone, UST cells only)\n" + "\n".join(text_ledger) + "\n"
```
(and change the rationale's "What the text corpus contributed: Nothing" wording.)

`Dockerfile`: add `COPY textlayer /opt/textlayer` next to `COPY t2agent /opt/t2agent`.

`submission.json` `models[]` (then reseal the descriptor digest):
```json
{"name": "nvidia/nemotron-3-super-120b-a12b", "version": "rl-030326-fp8", "revision": "rl-030326-fp8",
 "training_cutoff": "unpublished", "access": "api"}
{"name": "team304/ust-fed-tone-logit", "version": "v1", "revision": "sha256:<sha256 of textlayer/coefs.json>",
 "training_cutoff": "2024-12-18", "access": "local"}
```

`ARTIFACT_PROVENANCE.md`: add the layer (House model; fitted logistic in `coefs.json` with the training data
above; FOMC statements from federalreserve.gov, public domain, used offline only), and replace "the text corpus is
not read and the House model is not called".

Verified with the patch applied in a scratch copy: all 103 practice units pass g0–g3 offline (outputs identical
to the unpatched engine) and all 43 UST units pass with the stand-in endpoint live.

## Tests
```bash
cd submission && python textlayer/tests/test_textlayer.py
```
Failure modes (no endpoint, 401, refused, garbage, prose, no logprobs, no statement, budget) and invariants
(UST-only, ≤ 0.10 sd, width unchanged, finite, prompt identity).
