# Per-agent budgets and guardrail audit logging

**Status:** Accepted
**Date:** 2026-05-11

## Context

Two operational concerns surfaced after DEC-0011 and DEC-0012 expanded the
LLM surface area:

1. **Cost runaway.** A misbehaving prompt or an unbounded ReAct loop could
   issue tens of tool calls in a single agent invocation. The system had
   no per-agent budget cap.
2. **Guardrail observability.** Guardrails are firing all over — input
   validation, jurisdictional retrieval block, Phase-4 self-check, Council
   VETO, confidence hard caps. When something downgrades a confidence
   score or VETOs a quote, the customer-facing explanation should
   reference which guardrail fired. The DecisionTrace had this
   information but it wasn't easy to surface.

## Decision

We introduce **per-agent budgets** and **guardrail audit logging**:

### Per-agent budgets

Each agent has a budget tuple in `configs/llm_roles.yaml`:

```yaml
statutory_agent:
  model: openai:gpt-4o
  max_llm_calls: 3       # Phase-2 ReAct + Phase-3 emission + 1 retry
  max_tool_calls: 8      # ReAct recursion limit (24/3 = 8 iterations)
  max_tokens_per_call: 4096

risk_agent:
  model: openai:gpt-4o-mini
  max_llm_calls: 1       # one structured-output call
  max_tool_calls: 5      # 3-5 hazard probes
  ...
```

A budget breach is a soft failure: the agent stops early, writes a
`[BUDGET-BREACH]` audit node, and lets the safety-net fallback (where
applicable) take over. The pipeline never hard-fails on budget.

### Guardrail audit logging

Every guardrail that fires writes an audit node to the DecisionTrace with
a standardized payload:

```python
{
  "agent": "ConfidenceAggregator",
  "guardrail": "hard_cap_statutory_compliance",
  "fired": True,
  "details": "statutory_violations != [] → cap confidence at 0.5",
  "evidence_ids": [...],
}
```

The customer-facing `warnings[]` array surfaces the user-relevant subset of
these. The full audit log lives in DecisionTrace for the Validator and any
external auditor.

## Consequences

**Positive**
- Cost runaway is bounded per-agent, not per-pipeline. A single ReAct loop
  cannot consume the whole budget.
- Every guardrail firing is traceable. A regulator asking "which guardrails
  fired on this quote?" gets a single grep against DecisionTrace.
- Customer-facing warnings stay short while the full audit chain is
  preserved.

**Negative**
- One more YAML config to maintain. Mitigated by sane defaults that work
  for the two demo profiles.
- The budget enforcement code is per-agent; agents written without budget
  awareness will need to be updated.

## Verification

`tests/verify_replacements.py` extended to assert:
- StatutoryAgent budget breach forces Phase-5 fallback (test seam).
- RiskAgent budget breach forces deterministic-factor fallback.
- Every guardrail-fired audit node appears in DecisionTrace when
  conditions are met.
- The `warnings[]` array length corresponds to the user-relevant subset
  of fired guardrails.

## Alternatives considered

- Single global budget — rejected because per-agent visibility is more
  useful for cost analysis.
- No budgets, monitor in production — rejected because the demo runs
  hit a wide variety of profiles; a misbehaving prompt could blow the
  cost budget on a single demo.
