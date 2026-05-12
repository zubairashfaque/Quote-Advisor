"""Orchestration helper nodes (non-agent).

  - ``statutory_gate_node``  - applies the StatutoryRulesEngine pre-LLM
  - ``confidence_node``      - 4-signal deterministic aggregator (DEC-0004, v1.0)
  - ``output_assembler_node``- builds the final QuoteOutput
  - ``refusal_node``         - out-of-scope refusal path
  - routing helpers          - ``route_intent``, ``route_after_validator``, etc.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import Send

from . import statutory_rules_engine
from .agents.statutory_agent import statutory_agent_node
from .confidence import compute_confidence
from .decision_trace import append_node, make_node
from .schemas import (
    ConfidenceBreakdown,
    EligibilityRoute,
    IntentLabel,
    PremiumRange,
    QuoteOutput,
    Severity,
)


# =============================================================================
# Pre-LLM gate
# =============================================================================


def statutory_gate_node(state: dict[str, Any]) -> dict[str, Any]:
    """Statutory gate — delegates to the LLM-driven StatutoryAgent (DEC-0011).

    The agent retrieves statute prose from RAG and emits the same 8-field
    StatutoryEngineOutput that downstream agents consume. On any failure
    (LLM error, RAG outage, low grounding) the agent itself falls back to
    statutory_rules_engine.apply() — pipeline never ships malformed output.

    The deterministic engine is kept importable as the safety-net path
    inside statutory_agent_node (Phase 5).
    """
    return statutory_agent_node(state)


# =============================================================================
# Pricing parallel dispatch
# =============================================================================


def pricing_dispatch_router(state: dict[str, Any]) -> list[Send]:
    """Fan out the ReWOO Pricing Workers via LangGraph Send."""
    plan = state.get("pricing_plan") or []
    return [Send("pricing_worker", task) for task in plan]


# =============================================================================
# Confidence
# =============================================================================


def confidence_node(state: dict[str, Any]) -> dict[str, Any]:
    """Compute the 4-signal aggregate confidence (DEC-0004, v1.0).

    Deterministic compute_confidence runs first and fixes the number. The
    confidence_explainer (DEC-0012, augmentative LLM rationale paragraph) is
    documented as v1.1 work — see README §17. When wired, it cannot modify
    the number, the per-dimension scores, the hard cap, or the floors/ceilings.
    On explainer failure, rationale_summary is None and the pipeline continues.
    """
    from .agents.confidence_explainer import explain  # local import to avoid cycles

    breakdown: ConfidenceBreakdown = compute_confidence(state)
    breakdown.rationale_summary = explain(breakdown, state)

    node = make_node(
        agent="ConfidenceAggregator",
        summary=f"confidence_overall={breakdown.overall:.2f} (council_invoked={breakdown.council_invoked}).",
        payload={
            "breakdown":             breakdown.model_dump(),
            "explainer_invoked":     breakdown.rationale_summary is not None,
        },
    )
    return {
        "confidence_overall":  breakdown.overall,
        "confidence_breakdown": breakdown,
        "decision_trace":      append_node(list(state.get("decision_trace", [])), node),
    }


# =============================================================================
# Output assembler - produces the structured QuoteOutput
# =============================================================================


def _strip_risk_factor(rf: Any) -> dict[str, str]:
    if hasattr(rf, "model_dump"):
        d = rf.model_dump()
    else:
        d = dict(rf)
    return {
        "factor":     d.get("factor", ""),
        "severity":   d.get("severity") if isinstance(d.get("severity"), str) else getattr(d.get("severity"), "value", "low"),
        "rationale":  d.get("rationale", ""),
    }


def _strip_coverage(cv: Any) -> dict[str, str]:
    if hasattr(cv, "model_dump"):
        d = cv.model_dump()
    else:
        d = dict(cv)
    return {
        "type":      d.get("type", ""),
        "limit":     str(d.get("limit", "")),
        "rationale": d.get("rationale", ""),
    }


def output_assembler_node(state: dict[str, Any]) -> dict[str, Any]:
    """Build the final QuoteOutput in the public output contract shape."""
    risk_factors = [_strip_risk_factor(rf) for rf in (state.get("risk_factors") or [])]
    recommended_coverages = [_strip_coverage(cv) for cv in (state.get("recommended_coverages") or [])]

    premium = state.get("premium_range")
    if isinstance(premium, dict):
        premium = PremiumRange.model_validate(premium)
    if premium is None:
        premium = PremiumRange(low=0.0, high=0.0, currency="USD")

    explanation_lines: list[str] = []
    factor_chain = state.get("pricing_factor_chain") or []
    if factor_chain:
        explanation_lines.append(
            "Premium composed from: "
            + " × ".join(
                f"{e.multiplier if hasattr(e, 'multiplier') else e.get('multiplier'):.2f} ({e.name if hasattr(e, 'name') else e.get('name')})"
                for e in factor_chain[:6]
            )
            + "."
        )
    triggered = state.get("triggered_rules") or []
    if triggered:
        rule_ids = [r.rule_id if hasattr(r, "rule_id") else r.get("rule_id") for r in triggered]
        explanation_lines.append(f"Statutory rules applied: {', '.join(rule_ids)}.")
    council_verdict = state.get("council_verdict")
    if council_verdict:
        cv = council_verdict if hasattr(council_verdict, "model_dump") else council_verdict
        kind = cv.consensus_kind if hasattr(cv, "consensus_kind") else cv.get("consensus_kind")
        if kind:
            explanation_lines.append(f"Council convened: verdict={kind}.")

    explanation = " ".join(explanation_lines) or "Quote generated by the multi-agent advisor."

    confidence = state.get("confidence_overall") or 0.5
    warnings = list(state.get("warnings") or [])
    breakdown = state.get("confidence_breakdown")
    if isinstance(breakdown, dict):
        breakdown = ConfidenceBreakdown.model_validate(breakdown)

    output = QuoteOutput(
        risk_factors=risk_factors,
        recommended_coverages=recommended_coverages,
        premium_range=premium,
        explanation=explanation,
        confidence_score=round(float(confidence), 3),
        warnings=warnings,
        confidence_breakdown=breakdown,
        market_route=state.get("market_route"),
        refer_to_human=bool(state.get("refer_to_human") or False),
        factor_chain=list(state.get("pricing_factor_chain") or []),
        triggered_rules=list(state.get("triggered_rules") or []),
        counterfactual=state.get("counterfactual_delta"),
        decision_trace=list(state.get("decision_trace") or []),
    )
    return {"output": output}


# =============================================================================
# Refusal path
# =============================================================================


def refusal_node(state: dict[str, Any]) -> dict[str, Any]:
    """Out-of-scope refusal: produce a minimal QuoteOutput-shaped response with confidence floor."""
    output = QuoteOutput(
        risk_factors=[],
        recommended_coverages=[],
        premium_range=PremiumRange(low=0.0, high=0.0, currency="USD"),
        explanation="The Quote Advisor only handles questions about home insurance quotes for properties in CA or FL. Please rephrase your question.",
        confidence_score=0.05,
        warnings=["intent=out_of_scope; refusal path"],
    )
    return {"output": output}


# =============================================================================
# Routing helpers
# =============================================================================


def route_intent(state: dict[str, Any]) -> str:
    intent = state.get("intent")
    if isinstance(intent, str):
        intent = IntentLabel(intent)
    if intent == IntentLabel.OUT_OF_SCOPE:
        return "refusal"
    if intent == IntentLabel.EXPLANATION:
        return "followup"
    if intent == IntentLabel.COUNTERFACTUAL:
        return "counterfactual"
    return "statutory_gate"


def route_after_validator(state: dict[str, Any]) -> str:
    return "council" if state.get("council_invoked") else "confidence"
