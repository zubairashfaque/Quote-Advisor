"""Graph state types (TypedDict).

Per DEC-0002 the LangGraph graph state is a TypedDict (not Pydantic) so that
partial updates and reducer semantics work cleanly. Pydantic is used only at
the input/output boundary (see schemas.py).

InputState is what the CLI hands to ``graph.invoke``; GraphState is the
in-flight state that every node reads/writes; OutputState is the projection
the CLI prints.

Two list-valued fields use ``operator.add`` reducers so that:
  - ``decision_trace`` accumulates across sequential nodes' partial updates
    instead of being overwritten (each node returns its full new trace, but
    we want appends to merge).
  - ``pricing_results`` accumulates across the parallel Send-dispatched
    pricing workers (each worker returns one result; the reducer concatenates).

Other list fields (warnings, consistency_flags, etc.) are written by single
nodes only, so default overwrite semantics are fine.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages

from .schemas import (
    ConfidenceBreakdown,
    CounterfactualDelta,
    CouncilVerdict,
    DecisionNode,
    EligibilityRoute,
    FactorChainEntry,
    IntentLabel,
    PremiumRange,
    QuoteOutput,
    RecommendedCoverage,
    RiskFactor,
    RuleFire,
)


# =============================================================================
# Custom reducers
# =============================================================================


def _decision_trace_reducer(
    left: list[DecisionNode] | None,
    right: list[DecisionNode] | None,
) -> list[DecisionNode]:
    """Append-only EXCEPT when the right-hand fragment starts with a node whose
    payload carries ``_reset_trace == True`` — then it replaces ``left`` entirely.

    The Intent Classifier emits this marker on every NEW_QUOTE classification
    so that a fresh quote on a thread that already has a persisted trace
    starts with a clean slate. Follow-up and counterfactual paths leave the
    marker off, so they continue to extend the prior trace (which is what the
    Follow-up agent needs to walk).
    """
    if right and getattr(right[0], "payload", None) and right[0].payload.get("_reset_trace"):
        return list(right)
    return (list(left) if left else []) + (list(right) if right else [])


# =============================================================================
# Boundary states
# =============================================================================


class InputState(TypedDict, total=False):
    """What the CLI hands to ``graph.invoke``."""

    raw_profile: dict[str, Any]
    thread_id: str
    followup_question: str | None
    seed: int | None


class OutputState(TypedDict, total=False):
    """The projection the CLI prints (a subset of GraphState)."""

    output: QuoteOutput
    confidence_overall: float
    warnings: list[str]
    decision_trace: list[DecisionNode]


# =============================================================================
# In-flight working memory (single graph run)
# =============================================================================


class GraphState(TypedDict, total=False):
    """Shared graph state - all 8 agents and the orchestrator read/write here.

    Keys are documented inline. ``total=False`` so partial updates are valid.
    """

    # ---- Input boundary ------------------------------------------------------
    raw_profile: dict[str, Any]
    thread_id: str
    followup_question: str | None

    # ---- Pre-LLM gate output -------------------------------------------------
    sanitized_profile: dict[str, Any] | None
    triggered_rules: list[RuleFire]
    statutory_violations: list[str]
    field_treatments: dict[str, str]
    required_offers: list[dict[str, Any]]
    required_coverages: list[str]
    required_forms: list[str]
    floors: dict[str, Any]
    market_route_hints: list[str]

    # ---- Routing -------------------------------------------------------------
    intent: IntentLabel | None
    market_route: EligibilityRoute | None

    # ---- Per-agent outputs ---------------------------------------------------
    risk_factors: list[RiskFactor]
    recommended_coverages: list[RecommendedCoverage]
    premium_range: PremiumRange | None
    pricing_factor_chain: list[FactorChainEntry]

    # ---- Validator + Council -------------------------------------------------
    consistency_flags: list[str]
    council_invoked: bool
    council_verdict: CouncilVerdict | None

    # ---- Audit backbone (reducer: append, with NEW_QUOTE reset semantics) ---
    # Replaces the prior trace when the right-hand fragment's first node carries
    # ``payload['_reset_trace'] = True``. The Intent Classifier emits this marker
    # on every NEW_QUOTE classification so a fresh quote starts with a clean
    # trace instead of compounding the prior run's nodes (which would happen
    # under plain ``operator.add`` because MemorySaver persists state across
    # runs on the same ``thread_id``).
    decision_trace: Annotated[list[DecisionNode], _decision_trace_reducer]

    # ---- ReWOO parallel pricing results (reducer: append) -------------------
    pricing_plan: list[dict[str, Any]]
    pricing_results: Annotated[list[dict[str, Any]], operator.add]

    # ---- Confidence ----------------------------------------------------------
    confidence_overall: float | None
    confidence_breakdown: ConfidenceBreakdown | None

    # ---- Conversation --------------------------------------------------------
    messages: Annotated[list, add_messages]

    # ---- Counterfactual ------------------------------------------------------
    counterfactual_base_state: dict[str, Any] | None
    counterfactual_delta: CounterfactualDelta | None
    counterfactual_reflexion_memory: list[str]  # accumulates across turns in a thread

    # ---- Output assembly -----------------------------------------------------
    warnings: list[str]
    refer_to_human: bool
    output: QuoteOutput | None

    # ---- Telemetry -----------------------------------------------------------
    cost_usd: float
    tokens_in: int
    tokens_out: int
    retry_counts: dict[str, int]  # node_name -> retry count
