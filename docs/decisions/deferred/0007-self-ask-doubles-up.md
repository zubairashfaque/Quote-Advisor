# DEC-0007 · Self-Ask doubles up across Intent + Follow-up

**Status:** Accepted *(introduced in V5)*
**Date:** 2026-05-09

## Context

V4 originally assigned a unique cognitive pattern to each of the 8 agents
(Intent=Self-Ask, Eligibility=Tree-of-Thoughts, Risk=ReAct,
Coverage=Plan-and-Execute, Pricing=ReWOO, Validator+Council=Critic-Refine,
Counterfactual=Reflexion, Follow-up=???). The V4→V5 deliberation needed to
pick a pattern for Follow-up.

Both Follow-up Explain and the existing Intent Classifier have the same
underlying shape: **decompose a free-text question into ordered binary
sub-questions and answer them**. The Intent Classifier asks "is this a new
quote? a follow-up? a counterfactual? out of scope?" — four ordered binary
questions. Follow-up Explain asks "what is the top driver? what is the
second driver? does the question reference a corpus? is the answer
quantitative?" — same shape.

We considered three options:

1. Give Follow-up its own pattern (something exotic like ReWOO Lite or
   Critic-Refine) — would inflate the pattern count to 8 patterns for 8
   agents.
2. Reuse Self-Ask for Follow-up — pattern count stays at 7. Self-Ask runs
   twice in the system (once at the front door for Intent, once at the
   back door for Follow-up).
3. Collapse Intent and Follow-up into one agent — rejected because they
   have different cost profiles and routing semantics; merging them would
   muddy LangGraph Studio observability (per DEC-0006).

## Decision

**Self-Ask is used in two places: Intent Classifier (entry) and Follow-up
Explain (back-door explanation lane). The doubling is intentional.**

This keeps the cognitive-pattern count at **7 distinct patterns** for the
8 named agents:

| Agent | Pattern |
|---|---|
| IntentClassifier | Self-Ask |
| StatutoryAgent | ReAct + RAG (per DEC-0011) |
| EligibilityTriage | Tree-of-Thoughts |
| RiskAgent | ReAct |
| CoverageAgent | Plan-and-Execute |
| PricingAgent | ReWOO (Planner / Workers / Solver) |
| Validator + Council | Critic-Refine (4 personas) |
| Counterfactual | Reflexion |
| Follow-up Explain | Self-Ask *(reuse)* |

## Consequences

**Positive**
- Seven cognitive patterns covers the design space cleanly.
- Self-Ask reuse is well-motivated: both seats benefit from the
  decompose-then-answer shape and from the trace-friendly sub-question
  log that LangSmith captures.
- Adding a new "decompose-then-answer" agent in the future doesn't require
  inventing a new pattern.

**Negative**
- Two distinct prompts to maintain (`INTENT_CLASSIFIER` and
  `FOLLOWUP_EXPLAIN` in `prompts.py`). They are not deduplicated because
  the sub-question lists are different.
- A reader has to look twice to understand "why two Self-Asks?" — this
  DEC is the answer.

## Alternatives considered

- Eight unique patterns for eight agents — rejected for pattern-bloat.
- Collapse Intent and Follow-up — rejected for observability and routing
  cleanliness.
