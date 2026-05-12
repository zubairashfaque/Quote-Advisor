# DEC-0010 · Multi-axis Counterfactual via Reflexion ⊕ Tree-of-Thoughts

**Status:** Accepted
**Date:** 2026-05-09

## Context

The Counterfactual agent supports both **single-axis** mutations
(*"What if I removed the pool?"*) and **multi-axis** mutations
(*"What if I removed the pool AND raised my deductible AND added wind
mitigation?"*). The multi-axis case raises a question: **how should the
agent explore the mutation space?**

Three approaches were considered:

1. **Sequential apply** — apply mutations one at a time, re-running the
   pipeline between each. Pros: simple, observable. Cons: doesn't show
   interaction effects until the final result.
2. **Joint apply with Reflexion** — apply all mutations simultaneously to
   a forked state, re-run Risk → Coverage → Pricing once, use Reflexion
   memory to catch implausibility. Pros: shows the joint effect directly.
   Cons: no comparison of which mutation drove which part of the delta.
3. **Tree-of-Thoughts over mutation subsets** — explore the powerset of
   mutations as branches, score each branch's delta, surface the dominant
   contributors. Pros: gives "if you can only do one thing, do this".
   Cons: 2^n branches; expensive for n > 4.

## Decision

We use a **hybrid: Reflexion ⊕ Tree-of-Thoughts**.

- **Reflexion** is the outer loop for plausibility recovery (per
  DEC-0009).
- **Tree-of-Thoughts** is invoked when `len(mutation_axes) >= 2`. The
  agent expands a small tree:
  - Root: full joint mutation (all axes applied).
  - Children: subset mutations (each individual axis, plus pairs if
    n ≤ 4).
  - Leaves: deltas vs the base quote.
- The agent surfaces in the response:
  - The joint delta (root).
  - The per-axis contribution (single-axis children).
  - A "dominant driver" hint when one axis explains > 70% of the joint
    delta.

For single-axis mutations (n=1), no Tree-of-Thoughts is needed — the
agent just applies the mutation and re-runs (linear path).

## Consequences

**Positive**
- Multi-axis questions get a useful decomposition: customers asking *"what
  combination of changes would lower my premium most?"* get a ranked answer.
- The Tree-of-Thoughts branches are persisted in the DecisionTrace so a
  reviewer can see which subsets were explored.
- Reflexion still catches double-counting and implausibility on the joint
  delta.

**Negative**
- 2^n branches for n axes. We cap at 4 axes; beyond that the agent reports
  the joint result only.
- Slightly more expensive than a pure joint-apply approach; mitigated by
  parallel branch evaluation.

## Alternatives considered

- Sequential apply — rejected because joint effects matter for the
  customer's question.
- Joint-only — rejected because customers want to know per-axis
  contribution.
- Full powerset — rejected for cost; subset cap at pairs is enough for
  the demo's two-axis case.
