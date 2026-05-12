"""Confidence aggregator (DEC-0004).

Pure-Python 4-signal weighted aggregator. NEVER an LLM self-rating.

  0.25 · validation_pass_rate         no statutory violations + no retries
  0.25 · statutory_compliance         1.0 if all rules cleared, hard-cap at 0.5 if any violated
  0.25 · grounding_score              % regulatory/multiplier claims with evidence_id
  0.25 · input_completeness           1 - (missing/total) with FL credit-null docked lightly

Calibration:
  floor=0.05, ceiling=0.95, hard cap at 0.5 if statutory_compliance<1.0.

The previous 8-signal aggregator (agent_agreement_signal, retry_inverse,
numeric_consistency_score, council_agreement) is documented as deferred work
in README §16 Future Enhancements.
"""

from __future__ import annotations

from typing import Any

from .schemas import ConfidenceBreakdown

WEIGHTS = {
    "validation_pass_rate":       0.25,
    "statutory_compliance":       0.25,
    "grounding_score":            0.25,
    "input_completeness":         0.25,
}


def _f(x: float) -> float:
    """Clamp to [0,1]."""
    return max(0.0, min(1.0, float(x)))


def _statutory_compliance(state: dict[str, Any]) -> float:
    return 0.0 if (state.get("statutory_violations") or []) else 1.0


_META_AGENTS: set[str] = {
    # Orchestration nodes that record decisions but never make groundable claims.
    # Counting them in the denominator artificially deflates the grounding signal
    # because they have no evidence to cite. The Pricing/Risk/Coverage/Statutory
    # agents (which DO make claims with evidence_ids) are kept in the denominator.
    "IntentClassifier",
    "EligibilityTriage",
    "PricingPlanner",
    "Validator",
    "Council",
    "ConfidenceAggregator",
    "FollowupExplain",
}


def _grounding_score(state: dict[str, Any]) -> float:
    """Fraction of substantive decision-trace nodes that carry at least one evidence_id."""
    trace = state.get("decision_trace") or []
    if not trace:
        return 0.5
    grounded = 0
    total = 0
    for n in trace:
        agent = n.agent if hasattr(n, "agent") else (n or {}).get("agent", "")
        if agent in _META_AGENTS:
            continue
        evidence = n.evidence_ids if hasattr(n, "evidence_ids") else (n or {}).get("evidence_ids", [])
        total += 1
        if evidence:
            grounded += 1
    return grounded / total if total else 0.5


def _input_completeness(state: dict[str, Any]) -> float:
    sanitized = state.get("sanitized_profile") or {}
    required = ["age", "location", "home_value", "has_pool", "claims_history"]
    present = sum(1 for k in required if sanitized.get(k) is not None)
    base = present / len(required)
    # FL null credit -> dock 0.10 only (statutory protection, not error).
    treatments = state.get("field_treatments") or {}
    if treatments.get("credit_score", "").startswith("neutral"):
        base = max(0.0, base - 0.10)
    return _f(base)


def _validation_pass_rate(state: dict[str, Any]) -> float:
    """1.0 if no statutory_violations and no retries; decays with retry count."""
    sv = state.get("statutory_violations") or []
    retries = state.get("retry_counts") or {}
    if sv:
        return 0.0
    if retries:
        return _f(1.0 - 0.20 * sum(retries.values()))
    return 1.0


# =============================================================================
# Public API
# =============================================================================


def compute_confidence(state: dict[str, Any]) -> ConfidenceBreakdown:
    signals = {
        "validation_pass_rate":   _f(_validation_pass_rate(state)),
        "statutory_compliance":   _f(_statutory_compliance(state)),
        "grounding_score":        _f(_grounding_score(state)),
        "input_completeness":     _f(_input_completeness(state)),
    }
    overall = sum(signals[k] * w for k, w in WEIGHTS.items())

    # Calibration: floor 0.05, ceiling 0.95.
    overall = max(0.05, min(0.95, overall))

    # Hard cap at 0.5 if statutory_compliance < 1.0.
    if signals["statutory_compliance"] < 1.0:
        overall = min(overall, 0.5)

    # Per-dimension breakdown derived from the 4 signals.
    risk_dim = signals["grounding_score"]
    coverage_dim = signals["input_completeness"]
    pricing_dim = signals["validation_pass_rate"]
    grounding_dim = signals["grounding_score"]

    return ConfidenceBreakdown(
        overall=round(overall, 3),
        risk=round(_f(risk_dim), 3),
        coverage=round(_f(coverage_dim), 3),
        pricing=round(_f(pricing_dim), 3),
        grounding=round(_f(grounding_dim), 3),
        council_invoked=bool(state.get("council_invoked")),
        council_verdict=(
            (state.get("council_verdict").consensus_kind if hasattr(state.get("council_verdict"), "consensus_kind") else (state.get("council_verdict") or {}).get("consensus_kind"))
            if state.get("council_verdict") else None
        ),
    )
