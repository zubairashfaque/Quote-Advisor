# DEC-0005 · StatutoryRulesEngine as a hard pre-LLM gate

**Status:** Superseded by DEC-0011 *(the deterministic engine is kept as the
Phase-5 safety-net fallback within DEC-0011's 5-phase pipeline; it no longer
sits at the gate position)*
**Date:** 2026-05-09 (original); superseded 2026-05-10

## Context

The original V4 design treated statutory compliance as a **pre-LLM hard
gate**. A pure-Python `statutory_rules_engine.py` evaluated all 14 known
rules against the raw customer profile and emitted the 8-field
`StatutoryEngineOutput` before any LLM-driven agent ran. The rationale was
that statute violation is the highest-cost failure mode the system could
produce, and a deterministic engine guarantees the same input always yields
the same statutory output.

## Decision (original — superseded)

Run the deterministic `statutory_rules_engine.apply(profile)` first in the
pipeline. Its output (`sanitized_profile`, `triggered_rules`,
`field_treatments`, `required_offers`, `required_coverages`, `floors`,
`market_route_hints`, `statutory_violations`) flows into every downstream
agent. No LLM call participates in statutory evaluation.

## Why this was superseded (DEC-0011)

A hardcoded rule engine has two problems that became material:

1. **Statute updates require code changes.** A new CA insurance bulletin or
   a FL OIR amendment means a Python diff and a release. The rules engine
   doesn't read the statute prose at runtime — the prose is encoded in
   Python `if` statements.
2. **The audit chain is weaker.** A rule like `CA-PROP103-CREDIT` fires
   because Python code says so, not because the statute prose was actually
   retrieved and cited. The `evidence_id` is a literal string in code, not
   a chunk in a RAG corpus that anyone can read and verify.

DEC-0011 replaces the gate position with a ReAct + RAG agent that retrieves
statute prose from the per-jurisdiction corpora at runtime. The legacy
engine is **kept as the Phase-5 safety-net fallback** — it fires only on
LLM failure, malformed output, or low grounding. This DEC is preserved so
the rationale for the original choice (and the reason it was upgraded) is
visible in the project's decision history.

## Consequences of the original decision (now historical)

**Positive (what kept us going for a while)**
- Statute violation was structurally impossible while the engine sat at the
  gate position. Same input → same statutory output, always.
- The downstream agents could trust `sanitized_profile` and
  `field_treatments` without re-validating.

**Negative (what drove the supersession)**
- Statute updates required code changes.
- The audit chain bottomed out in Python source, not in the regulator's
  published prose.

## Alternatives that were considered at the time

- LLM-only statutory evaluation — rejected because of consistency concerns
  (until DEC-0011 introduced the safety-net pattern).
- LLM + post-hoc validation — rejected because two LLM calls per quote is
  more expensive than the deterministic-first design.
