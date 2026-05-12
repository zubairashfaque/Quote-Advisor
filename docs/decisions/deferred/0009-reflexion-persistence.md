# DEC-0009 · Reflexion memory persists across turns within a thread

**Status:** Accepted
**Date:** 2026-05-09

## Context

The Counterfactual agent uses the **Reflexion** pattern: on a plausibility
failure (e.g., the fork's premium delta is implausibly large or implausibly
small relative to the cohort band), the agent writes a short verbal
self-critique into a memory slot, and re-runs the solver with the critique
in context. The pattern is "trial → reflect → re-trial" with the reflection
informing the next attempt.

The question was: **what is the scope of the reflexion memory?**

Three options:

1. **Per-call** — memory exists only inside one Counterfactual invocation;
   cleared when the call returns.
2. **Per-thread (within a `thread_id`)** — memory persists across turns on
   the same thread but is isolated across different `thread_id`s.
3. **Global** — memory persists across all customers and all sessions.

## Decision

Reflexion memory **persists across turns within a single `thread_id`**.

The memory slot lives in `GraphState` as `counterfactual_reflexion_memory:
list[str]`. When the Counterfactual agent runs:
- On entry, it reads the current memory list (may be empty on first turn).
- After a plausibility failure, it appends a 1-sentence verbal reflection
  to the list.
- The reflection becomes context for the next solver call within the same
  thread.

Because `GraphState` is checkpointed by LangGraph's `MemorySaver` keyed by
`thread_id`, the memory survives between user turns on the same thread but
is fresh on a different thread.

## Consequences

**Positive**
- Multi-turn counterfactual conversations get better as the agent learns
  from prior plausibility failures. ("Last time I double-counted the pool
  surcharge; next time I'll check.")
- Cross-customer leakage is structurally impossible. Each customer has
  their own `thread_id`; their reflexion memory is isolated.
- The memory is observable in LangSmith (the trace shows the reflection
  list in each Counterfactual call).

**Negative**
- Memory grows unboundedly within a long-running thread. Mitigated by:
  reflections are short (1-sentence), and most threads have ≤ 5 turns.
- Counterfactual must be deterministic-ish given the same memory + same
  input. Plausibility-failure conditions are themselves deterministic
  (cohort band check), so this holds.

## Alternatives considered

- Per-call memory — rejected because Reflexion's value compounds across
  turns; per-call resets the pattern's main benefit.
- Global memory — rejected for privacy reasons (one customer's reflection
  could subtly influence another customer's quote).
