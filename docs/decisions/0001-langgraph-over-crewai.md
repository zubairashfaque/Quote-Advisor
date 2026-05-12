# Use LangGraph as the multi-agent orchestrator

**Status:** Accepted
**Date:** 2026-05-09

## Context

The Quote Advisor pipeline is a multi-agent system with 14 nodes, four
conditional intent lanes, parallel ReWOO worker fan-out for the Pricing
Agent, and per-`thread_id` state persistence so Follow-up and Counterfactual
turns can re-enter without re-running the upstream pipeline. We evaluated
three candidate orchestrators:

- **LangGraph** — graph-as-code, TypedDict state, conditional edges, native
  `Send` primitive for parallel fan-out, MemorySaver for thread persistence,
  first-class LangSmith trace integration.
- **CrewAI** — role-based agent orchestration, less explicit graph control,
  weaker support for conditional routing and parallel fan-out.
- **LlamaIndex Workflows** — newer; primitives for conditional routing exist
  but parallel fan-out and durable state are less mature.
- **Hand-rolled SDK** — full control but every primitive (parallel worker
  dispatch, per-thread checkpoint, trace export) would have to be built.

## Decision

We use **LangGraph** for the orchestrator. The graph is built as a
`StateGraph` over a TypedDict `GraphState`; each agent is a Python function
that reads slices of state and returns slices of state; LangGraph merges the
returns using per-field reducers.

## Consequences

**Positive**
- Conditional edges off the IntentClassifier route based on `intent` value
  are first-class in the API — observable in LangGraph Studio.
- PricingPlanner's parallel worker fan-out uses `Send` directly with an
  `operator.add` reducer on `pricing_results`. No custom dispatch code.
- MemorySaver SQLite backend persists state by `thread_id` for free —
  Follow-up and Counterfactual just call `graph.invoke(config={"configurable":
  {"thread_id": "demo-a"}})` and the prior state is restored automatically.
- LangSmith tracing is wired via env vars; every ReAct trajectory, every tool
  call, every LLM input/output is recorded without any agent-side code.

**Negative**
- LangGraph is the heaviest of the candidates in terms of dependencies.
- The TypedDict state model means Pydantic validation only happens at the
  I/O boundary (see DEC-0002). Misuse risk: an agent could write a malformed
  slice into state.

## Alternatives considered

- CrewAI was rejected because its role-based abstraction is less explicit
  than a graph for conditional routing.
- LlamaIndex Workflows was rejected because its parallel fan-out primitives
  and durable-state story were less mature at evaluation time.
- Hand-rolled SDK was rejected because the orchestration code would have
  dominated the project's scope.
