# DEC-0012 · LLM Confidence Explainer augments DEC-0004 (writes prose only)

**Status:** Accepted *(augments DEC-0004; does not supersede it)*
**Date:** 2026-05-10

## Context

DEC-0004 established `confidence_score` as a deterministic 8-signal
weighted aggregator. The number is defensible: a regulator can reproduce
it from observable trace properties. But the customer-facing experience
needed something more — a **plain-language explanation** of what the
number means and what could change it.

The risk: if an LLM is allowed to write the explanation, it might
inadvertently change the number too (e.g., by re-evaluating the signals
in prose and arriving at a different conclusion that contradicts the
deterministic math).

## Decision

DEC-0004's deterministic compute is **unchanged**. After the number is
fixed, DEC-0012 adds one cheap LLM call — the **Confidence Explainer** —
that produces a 2-3 sentence `rationale_summary` paragraph.

The Explainer:
- **Receives** the deterministic `confidence_score`, the per-signal
  sub-scores, the firing hard caps, and a short profile summary.
- **Writes** 2-3 sentences naming:
  1. The dominant signal pushing confidence up or down.
  2. Any hard caps that fired (e.g., "statutory non-compliance → cap at
     0.5").
  3. What the customer could change in a future quote to lift the score
     (e.g., "providing a roof-age field would lift profile_completeness").
- **Cannot** modify `confidence_score`, the sub-scores, the cap, or any
  other numeric field. The Explainer's output type is a single string;
  the orchestrator writes it to `confidence_breakdown.rationale_summary`
  and the customer sees it via `QuoteOutput.confidence_breakdown`.

The contract is enforced at three layers:
1. **Prompt instruction.** The system prompt explicitly says "do not
   compute or recompute the number; write prose about the given number."
2. **Structured output type.** `with_structured_output(str)` — the LLM
   can only return a string.
3. **Code-side write isolation.** The Explainer's return is assigned to
   `rationale_summary` only; nothing reads it back into a numeric field.

## Consequences

**Positive**
- Customers get a paragraph in plain English explaining what their
  confidence score means and what they could change.
- The deterministic math is untouched. The regulator-defensibility from
  DEC-0004 is preserved.
- One cheap LLM call (~$0.0001 with `gpt-4o-mini`). Negligible cost.

**Negative**
- One more LLM call to monitor. Failure mode: explainer LLM raises →
  rationale_summary is empty, customer sees the number without prose.
  Acceptable degradation.
- The Explainer prompt must stay synchronized with the aggregator's
  signal list. If a new signal is added to `confidence.py`, the prompt
  should mention how to surface it.

## Verification

`tests/verify_replacements.py` asserts:
- `confidence_score` is identical before and after DEC-0012's introduction.
- `rationale_summary` is non-empty on both demo profiles.
- Mutating `rationale_summary` in test fixtures does not affect any
  numeric field.

## Alternatives considered

- Let the LLM write both the number and the prose — rejected because it
  undermines DEC-0004's defensibility.
- Hand-template the prose from sub-score signals — rejected because the
  resulting prose is robotic and lacks per-profile relevance.
