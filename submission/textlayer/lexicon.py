"""Traditional NLP on FOMC statements: hawkish/dovish phrase tone (used by the fallback model) and sentiment
readings (economic conditions, uncertainty, wording change) that are reported in the rationale only.

The phrase lists are written from domain knowledge, frozen, and byte-identical to the research ones the lexicon
coefficients were fitted with (research/nlp/backtest/step1_nlp.py, step1b_compare.py); a test fails if they drift.
The sentiment readings do NOT move the forecast: adding them to the model lowered out-of-time AUC (.68-.70 -> .58-.61).
"""

from __future__ import annotations

import re

HAWK = ["inflation pressures", "upside risk", "risks to inflation", "further firming", "additional firming",
        "policy firming", "ongoing increases", "further increases", "further gradual increases", "raise the target",
        "increase the target", "remain attentive to inflation", "inflation remains elevated", "elevated inflation",
        "highly attentive to inflation", "returning inflation to its 2 percent", "tight", "reduce its holdings",
        "reducing its holdings", "removal of policy accommodation", "measured pace", "strong", "robust",
        "heightened inflation", "inflation risks", "inflationary", "price pressures"]
DOVE = ["lower the target", "lowered the target", "reduce the target", "accommodative", "considerable period",
        "patient", "downside risks", "weaken", "deteriorat", "act as appropriate to sustain", "subdued",
        "below the committee's 2 percent", "extended period", "exceptionally low", "asset purchases", "reinvest",
        "slack", "strains", "turmoil", "disruption", "softening", "moderat", "uncertain", "decline", "contraction"]
ECON_POS = ["expand", "expansion", "strong", "strengthen", "solid", "robust", "improv", "gains", "rising", "pick up",
            "picked up", "firm", "healthy", "recover", "accelerat", "advanc", "brisk"]
ECON_NEG = ["weak", "declin", "deteriorat", "slow", "soft", "contraction", "contract", "recession", "decrease", "fell",
            "falling", "downturn", "strain", "stress", "turmoil", "losses", "sluggish", "subdued", "sagg", "faltering"]
UNCERT = ["uncertain", "risk", "unclear", "unpredictab", "volatil", "may ", "could ", "possib", "depend", "evolv",
          "monitor", "closely", "attentive"]


def tone(text: str) -> float:
    """(hawkish - dovish) / (hawkish + dovish + 1) phrase counts; positive = hawkish."""
    t = text.lower()
    h = sum(t.count(p) for p in HAWK)
    d = sum(t.count(p) for p in DOVE)
    return (h - d) / (h + d + 1)


def _words(t: str) -> list[str]:
    return re.findall(r"[a-z]+", t.lower())


def econ_sentiment(text: str) -> float:
    s = text.lower()
    p, n = sum(s.count(w) for w in ECON_POS), sum(s.count(w) for w in ECON_NEG)
    return (p - n) / (p + n + 1)


def uncertainty(text: str) -> float:
    s = text.lower()
    return 1000.0 * sum(s.count(w) for w in UNCERT) / max(len(_words(text)), 1)


def novelty(cur: str, prev: str | None) -> float | None:
    """1 - Jaccard similarity of word trigrams with the previous statement (large = rewritten statement)."""
    if not prev:
        return None
    tri = lambda t: {tuple(w[i:i + 3]) for w in [_words(t)] for i in range(len(w) - 2)}  # noqa: E731
    a, b = tri(cur), tri(prev)
    return 1.0 - len(a & b) / max(len(a | b), 1)
