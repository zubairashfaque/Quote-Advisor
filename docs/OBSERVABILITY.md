# Observability

How to trace, audit, and inspect a Quote Advisor run. Referenced from [README Section 14](../README.md#14-observability).

## 1. LangSmith trace URLs

With `--verbose`, the CLI prints `LangSmith trace: https://smith.langchain.com/projects/refocusai/traces?metadata=thread_id:demo-a` -- click through for the full agent/tool/LLM call tree.

## 2. `--verbose` DecisionTrace + guardrail-event dump

`--verbose` writes the full DecisionTrace to stderr (every node summary + `evidence_ids`), interleaved with `[GUARDRAIL ...]` lines -- one per guardrail invocation -- forming the regulator-greppable audit trail:

```
STEP-001  IntentClassifier      Initial profile classified as new_quote.
STEP-002  StatutoryAgent        ReAct agent fired 6 rule(s) (5 retrieval(s); 0 dropped by self-check). [RULE-CA-PROP103-CREDIT, ...]
STEP-003  EligibilityTriage     market_route=fair_dic: FHSZ Very High forces FAIR Plan + DIC route.
[GUARDRAIL input_validation] role=- event=passed reason=profile validated payload={"warnings_count":0}
[GUARDRAIL token_budget]     role=STATUTORY_AGENT event=passed reason=in-budget payload={"total_tokens":8431,"limit":15000}
[GUARDRAIL pii_scrubber]     role=- event=passed reason=no PII keys found payload={}
```

Event types: `passed`, `fired`, `fallback`, `abort`. Token-budget breaches with `action_on_breach=fallback` surface as `event=fallback` followed by the agent's safety-net node.

## 3. `--llm-trace` resolved model table

```
   Resolved per-agent LLM assignments
+-----------------------+--------------------------------+
| Role                  | Model                          |
+-----------------------+--------------------------------+
| COUNCIL_ACTUARY       | openai:gpt-4o                  |
| COUNCIL_ADVOCATE      | openai:gpt-4o                  |
| COUNCIL_COMPLIANCE    | openai:gpt-4o                  |
| COUNCIL_UNDERWRITER   | openai:gpt-4o                  |
| COUNTERFACTUAL        | openai:gpt-4o                  |
| COVERAGE_EXECUTOR     | openai:gpt-4o-mini             |
| COVERAGE_PLANNER      | openai:gpt-4o                  |
| ELIGIBILITY_TRIAGE    | openai:gpt-4o-mini             |
| FOLLOWUP_EXPLAIN      | openai:gpt-4o-mini             |
| INTENT_CLASSIFIER     | openai:gpt-4o-mini             |
| PRICING_PLANNER       | openai:gpt-4o                  |
| PRICING_SOLVER        | openai:gpt-4o                  |
| RISK_AGENT            | openai:gpt-4o                  |
| VALIDATOR             | openai:gpt-4o                  |
+-----------------------+--------------------------------+
```

## 4. Per-run telemetry schema

For the full per-run JSON-line record the system is designed to emit, see [`TELEMETRY_SCHEMA.md`](TELEMETRY_SCHEMA.md). LangSmith covers the live wire-up; the offline schema is the production-emit target.
