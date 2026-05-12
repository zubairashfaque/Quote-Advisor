# Per-agent LLM registry over single-model graph

**Status:** Accepted
**Date:** 2026-05-09 *(refined 2026-05-10 — OpenAI made default provider)*

## Context

The pipeline has **14 distinct LLM roles**:
- 8 named agents (Intent, Statutory, Eligibility, Risk, Coverage, Pricing,
  Validator, Counterfactual, Follow-up — 9 agents, but Validator+Council
  share the role count differently)
- 4 Council personas (Compliance, Actuarial Analyst, Customer Advocate,
  Risk Manager)
- Coverage has 2 sub-roles (Planner, Executor)
- Pricing has 2 sub-roles (Planner, Solver)

Different roles benefit from different model choices:
- Intent Classifier: small, fast, cheap (`gpt-4o-mini` is fine)
- StatutoryAgent ReAct loop: needs reasoning depth (`gpt-4o`)
- Pricing Solver: prose-only polish (`gpt-4o-mini`)
- Council Compliance persona: needs to push back authoritatively
  (`gpt-4o`)

The naïve approach — one model for the entire graph — overpays on cheap
roles and may underpay on hard roles.

## Decision

We use a **per-agent LLM registry** that resolves each role to a model
identifier via `llm_registry.get_llm(role)`:

```python
class AgentRole(str, Enum):
    INTENT_CLASSIFIER = "intent_classifier"
    STATUTORY_AGENT = "statutory_agent"
    ELIGIBILITY_TRIAGE = "eligibility_triage"
    RISK_AGENT = "risk_agent"
    COVERAGE_PLANNER = "coverage_planner"
    COVERAGE_EXECUTOR = "coverage_executor"
    PRICING_PLANNER = "pricing_planner"
    PRICING_SOLVER = "pricing_solver"
    VALIDATOR = "validator"
    COUNCIL_COMPLIANCE = "council_compliance"
    COUNCIL_ACTUARY = "council_actuary"
    COUNCIL_ADVOCATE = "council_advocate"
    COUNCIL_RISK = "council_risk"
    CONFIDENCE_EXPLAINER = "confidence_explainer"
    COUNTERFACTUAL = "counterfactual"
    FOLLOWUP_EXPLAIN = "followup_explain"
```

Resolution chain:
1. Per-role override via env var `QA_LLM_<ROLE>` (e.g.,
   `QA_LLM_STATUTORY_AGENT=openai:gpt-4o`)
2. Default mapping in `configs/llm_roles.yaml`
3. Project-wide fallback (`openai:gpt-4o-mini`)

All resolved through `init_chat_model("provider:model")` so the provider
is parameterized. **Agent code never instantiates a provider SDK
directly** — it always goes through the registry.

## Consequences

**Positive**
- Cost optimization: 11 of 16 roles use `gpt-4o-mini` (~$0.0001 each),
  5 use `gpt-4o` (~$0.005 each). Net cost per quote is much lower than a
  single-model graph using `gpt-4o` everywhere.
- A/B testing: swap one role's model via env var without touching code.
- Provider portability: changing from OpenAI to Anthropic for one role is
  a one-line config change.

**Negative**
- 16 LLM roles to track. Mitigated by the central `configs/llm_roles.yaml`
  registry — one file lists all defaults.
- Per-role calibration: a prompt that works at `gpt-4o-mini` may not work
  at `gpt-4o-mini-128k` even though the model family is similar.

## Alternatives considered

- Single-model graph — rejected for cost reasons.
- Hard-code model per agent in code — rejected for portability; env var
  + YAML allows runtime swap without rebuild.
