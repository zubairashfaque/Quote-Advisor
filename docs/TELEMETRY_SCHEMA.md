# Telemetry & Metrics Schema

Per-run JSON-line record (designed; not auto-emitted at runtime - LangSmith is the runtime sink). Every run produces a single record with the following shape; with `--verbose` the CLI prints a derived subset to stderr.

```json
{
  "run_id":           "uuid-v4",
  "thread_id":        "demo-a",
  "started_at":       "2026-05-09T19:00:00Z",
  "ended_at":         "2026-05-09T19:00:08Z",
  "is_followup":      false,
  "intent":           "new_quote",
  "market_route":     "fair_dic",
  "profile_kind":     "A",

  "node_timings_ms": {
    "intent":            320,
    "statutory_gate":      4,
    "eligibility":       280,
    "risk":             1840,
    "coverage":         2100,
    "pricing_planner":   480,
    "pricing_workers":  1620,
    "pricing_solver":    910,
    "validator":         240,
    "council":          3200,
    "confidence":         12,
    "output":              5
  },

  "tokens": {
    "input":  6840,
    "output": 1920,
    "total":  8760
  },

  "cost_usd_estimated": 0.142,

  "tools_called": [
    {"name": "fema_nri_risk",       "ok": true, "evidence_id": "NRI-CA-06037-2025"},
    {"name": "ca_fire_zone",        "ok": true, "evidence_id": "FHSZ-OBJ-12847-SRA-2024"},
    {"name": "usgs_seismic",        "ok": true, "evidence_id": "PGA-LAT34.05-LON-118.24-2026"},
    {"name": "lender_floor",        "ok": true, "evidence_id": "LENDER-FANNIE-B7-3-02-2024"},
    {"name": "base_premium",        "ok": true, "evidence_id": "BENCH-CA-2026-NEWSWEEK"},
    {"name": "pricing_multiplier_lookup", "ok": true, "evidence_id": "MULT-WF-VERY-HIGH"}
  ],

  "agents_invoked": [
    "IntentClassifier",
    "StatutoryRulesEngine",
    "EligibilityTriage",
    "RiskAgent",
    "CoverageAgent",
    "PricingPlanner",
    "PricingAgent",
    "Validator",
    "Council",
    "ConfidenceAggregator"
  ],

  "models_used": {
    "INTENT_CLASSIFIER":     "openai:gpt-4o-mini",
    "ELIGIBILITY_TRIAGE":    "openai:gpt-4o-mini",
    "RISK_AGENT":            "openai:gpt-4o",
    "COVERAGE_PLANNER":      "openai:gpt-4o",
    "COVERAGE_EXECUTOR":     "openai:gpt-4o-mini",
    "PRICING_PLANNER":       "openai:gpt-4o",
    "PRICING_SOLVER":        "openai:gpt-4o",
    "VALIDATOR":             "openai:gpt-4o",
    "COUNCIL_UNDERWRITER":   "openai:gpt-4o",
    "COUNCIL_ADVOCATE":      "openai:gpt-4o",
    "COUNCIL_ACTUARY":       "openai:gpt-4o",
    "COUNCIL_COMPLIANCE":    "openai:gpt-4o",
    "COUNTERFACTUAL":        "openai:gpt-4o",
    "FOLLOWUP_EXPLAIN":      "openai:gpt-4o-mini"
  },

  "outcome": {
    "premium_low_usd":   9000,
    "premium_high_usd":  14000,
    "confidence_overall": 0.78,
    "confidence_breakdown": {
      "risk":      0.92,
      "coverage":  0.85,
      "pricing":   0.71,
      "grounding": 0.80
    },
    "council_invoked":  true,
    "council_verdict":  "majority",
    "refer_to_human":   false,
    "warnings":         ["FAIR Plan likely required"]
  },

  "decision_trace_size": 17,
  "evidence_ids_unique": 14,
  "retries":             0,

  "langsmith_url": "https://smith.langchain.com/projects/refocusai/traces?metadata=thread_id:demo-a"
}
```

## Cost ledger derivation

`cost_usd_estimated` is computed at run-end from token counts × per-model rate from `configs/llm_costs.yaml` (not committed - operator-supplied). Default behaviour is to print an estimate but not enforce a cap; production would add a hard cost cutoff via the Token Budget guardrail.

## LangSmith integration

When `LANGSMITH_TRACING=true`, every LLM and Tool call is auto-traced. The trace is filterable by `metadata.thread_id`, which the CLI surfaces in the `--verbose` output. No agent code references LangSmith directly — `configuration._bootstrap_langsmith` populates env vars before `init_chat_model` constructs any provider client.

## Where this is emitted

- **At runtime (already wired):** LangSmith captures the agentic flow and all LLM/Tool calls; `--verbose` prints a derived subset to stderr.
- **Phase-2 (designed, not built):** a `telemetry/` JSONL file written from a graph hook that snapshots at `output_assembler_node`. The schema above is the target shape.
