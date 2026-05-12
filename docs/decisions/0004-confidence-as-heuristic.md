# Confidence as a deterministic 4-signal weighted aggregator

**Status:** Accepted *(v1.0 ships 4 signals; the original 8-signal design and
DEC-0012 LLM rationale paragraph are deferred to v1.1 — see README §17)*
**Date:** 2026-05-09 *(curated 2026-05-12)*

## Context

The output schema includes a `confidence_score ∈ [0.0, 1.0]` field that the
customer and any auditor will read. Three approaches were considered:

1. **Ask the LLM directly** — `llm.predict("on a scale of 0 to 1, how
   confident are you in this quote?")`. Simple but indefensible — the number
   is a self-assessment with no observable basis.
2. **Train a calibrated probability classifier** — collect labelled
   regulator-acceptable-vs-rejected quotes and fit a model. Defensible but
   requires labelled data the project does not have.
3. **Deterministic heuristic from observable trace properties** — weight a
   small set of signals that can be measured from the DecisionTrace and the
   pipeline output. Hand-chosen weights; calibrate against an eval set when
   one exists.

## Decision

We compute `confidence_score` as a **deterministic weighted sum of 4 signals**:

| Signal | Weight | Measures |
|---|---:|---|
| `validation_pass_rate` | 0.25 | No statutory violations and no retries; decays with retry count |
| `statutory_compliance` | 0.25 | `statutory_violations` list is empty (hard cap at 0.5 if violated) |
| `grounding_score` | 0.25 | % of substantive decision-trace nodes carrying an evidence_id |
| `input_completeness` | 0.25 | Fraction of required fields present; FL null-credit docked 0.10 |

Weights sum to 1.00. The raw weighted sum is bounded to `[0.05, 0.95]`. Then
the **hard cap** applies:

- `statutory_compliance < 1.0` → overall capped at 0.5

The original 8-signal design (adding `agent_agreement_signal`, `retry_inverse`,
`numeric_consistency_score`, `council_agreement`) is deferred to v1.1 — see
README §17. Likewise DEC-0012's augmentative LLM rationale paragraph is
deferred. The 4-signal version preserves the load-bearing properties
(grounding, statute, completeness, validation) at minimum surface area.

The math lives in `src/quote_advisor/confidence.py`. The LLM has zero ability
to change the number, the per-signal sub-scores, or the cap.

## Consequences

**Positive**
- Defensible to a regulator: the score is reproducible from observable trace
  properties.
- Hard caps prevent a high-scoring-on-other-signals quote from shipping at
  high confidence when statute is violated.
- Easy to extend: add a 9th signal, adjust weights, rebalance.

**Negative**
- Weights are hand-chosen. The 0.0–1.0 range is meaningful within the system
  but is not a literal calibrated probability.
- Production rollout should calibrate weights against an eval set of
  regulator-acceptable vs rejected quotes. That eval set does not exist yet.

## Alternatives considered

- Ask the LLM directly — rejected (indefensible, sycophant-prone).
- Train a probability classifier — deferred until eval data exists.
