# Dedicated Intent Classifier node, not self-routing

**Status:** Accepted
**Date:** 2026-05-09

## Context

When a request enters the graph, it must be routed to one of four lanes:
`NEW_QUOTE`, `EXPLANATION`, `COUNTERFACTUAL`, or `OUT_OF_SCOPE`. Two
implementation strategies were considered:

1. **Self-routing** — each lane's first agent (Follow-up, Counterfactual,
   etc.) decides whether it should handle the request and either processes
   it or hands off to the right sibling. The Follow-up agent often serves
   double-duty as the router.
2. **Dedicated Intent Classifier** — a separate node at the graph entrance
   that emits a labeled intent and a conditional edge routes from there.

## Decision

We use a **dedicated `IntentClassifier` node** at the graph entrance. It
runs Self-Ask decomposition (see DEC-0007) over four ordered binary
sub-questions and emits an enum-typed `IntentLabel`. A conditional edge
routes the graph based on `state.intent`.

The classifier also has two **deterministic short-circuit bypasses** that
skip the LLM entirely:
- profile present + no follow-up text → `NEW_QUOTE` (no LLM call)
- no profile + no follow-up text → `OUT_OF_SCOPE` (no LLM call)

## Consequences

**Positive**
- **Observable in LangGraph Studio.** The routing decision is a graph edge,
  not buried inside a sibling agent. A reviewer can see which lane fires
  for any input.
- **Counterfactual is directly invocable.** When mutation_axes are extracted
  from a follow-up question, the graph can route straight to the
  Counterfactual agent — no need to enter Follow-up and then re-route.
- **The four lanes have different cost profiles** (NEW_QUOTE ~$0.025,
  EXPLANATION ~$0.005, COUNTERFACTUAL ~$0.015, OUT_OF_SCOPE ~$0.001). A
  separate classifier makes the budget per lane visible.

**Negative**
- One extra LLM call per request (when the bypass doesn't fire). Mitigated
  by the bypasses, which cover the common cases.
- Two places to look when routing misbehaves: the classifier's
  Self-Ask trajectory + the conditional edge's predicate.

## Alternatives considered

- Have Follow-up self-route — rejected for observability reasons.
- Use a regex-based dispatcher — rejected because intent detection on free
  text is brittle without semantic understanding.
