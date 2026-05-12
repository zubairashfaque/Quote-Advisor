# DEC-0002 · TypedDict for graph state, Pydantic only at the I/O boundary

**Status:** Accepted
**Date:** 2026-05-09

## Context

LangGraph supports both TypedDict and Pydantic models for `GraphState`. The
choice affects developer ergonomics, runtime cost, and where validation
errors surface.

- **TypedDict** is a structural type hint. State slices are plain `dict`s at
  runtime. Reducers merge dicts cheaply. No per-write validation cost.
- **Pydantic** validates every write. Strong type safety end-to-end, but
  per-node validation adds overhead and pushes ValidationErrors into deep
  call stacks where they are harder to recover from.

## Decision

We use **TypedDict for `GraphState`** (the LangGraph state object that
flows between agents) and **Pydantic only at the I/O boundary** — the
incoming `CustomerProfile` (raw input → validated profile) and the outgoing
`QuoteOutput` (final structured response to the customer).

The boundary modules are:
- `src/quote_advisor/schemas.py` — Pydantic models: `CustomerProfile`,
  `QuoteOutput`, `RiskFactor`, `RecommendedCoverage`, `PremiumRange`, etc.
- `src/quote_advisor/state.py` — TypedDict `GraphState` with reducer
  annotations (`Annotated[list, operator.add]` for `pricing_results`,
  `Annotated[list, append_node]` for `decision_trace`, etc.).

## Consequences

**Positive**
- Cheap state merges. Each agent return is a dict; reducer is `dict.update`
  (default) or a per-field combinator. No per-write Pydantic validation cost.
- One place to look for validation failures: the boundary. Malformed customer
  input fails at `CustomerProfile.model_validate(raw)` with a clear error
  before any downstream agent runs.
- Output contract is enforced at the boundary too: a malformed agent return
  can be detected at `QuoteOutput.model_validate(state_slice)`.

**Negative**
- No type-safety guarantees mid-graph. An agent could write `state["intent"] =
  "blarg"` and the graph would happily carry that forward; only a downstream
  node consuming `intent` as an enum would fail.
- Mitigated by: every agent's structured output (the LLM call) uses
  `with_structured_output(SomePydanticModel)` so shape is enforced at the LLM
  SDK boundary, before it reaches the graph.

## Alternatives considered

- Full Pydantic state — rejected for the per-write cost and for the harder
  recovery path on mid-graph validation failures.
- Plain dict with no typing — rejected for lack of IDE/editor support.
