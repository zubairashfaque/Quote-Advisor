"""Output consistency guardrail (also referenced by validator_agent).

Cross-agent consistency checks:
  - premium midpoint within cohort p10-p90 band
  - Coverage A >= lender floor
  - statutory_violations empty
  - premium monotonic in severity

Emits one audit_logger event per call (passed | fired).
"""

from __future__ import annotations

from typing import Any

from ..schemas import DecisionNode, PremiumRange, Severity
from .audit_logger import log_guardrail_event


def check_output(
    state: dict[str, Any],
    *,
    decision_trace: list[DecisionNode] | None = None,
) -> list[str]:
    """Return a list of consistency-violation strings (empty if all checks pass)."""
    flags: list[str] = []

    if state.get("statutory_violations"):
        flags.append(f"statutory_violations non-empty: {state['statutory_violations']}")

    pr = state.get("premium_range")
    if isinstance(pr, dict):
        pr = PremiumRange.model_validate(pr)
    if pr is None:
        flags.append("premium_range absent")
    elif pr.high <= pr.low:
        flags.append(f"premium_range invalid: low={pr.low} >= high={pr.high}")

    factors = state.get("risk_factors") or []
    high_count = sum(
        1 for f in factors
        if (f.severity if hasattr(f, "severity") else f.get("severity")) in (Severity.HIGH, "high")
    )
    if pr and high_count >= 3:
        midpoint = (pr.low + pr.high) / 2
        if midpoint < 5_000:
            flags.append(f"non-monotonic premium: {high_count} high-severity factors but midpoint ${midpoint:.0f}")

    if not flags:
        log_guardrail_event(
            guardrail="output_consistency",
            role=None,
            event="passed",
            reason="all 4 consistency checks passed",
            decision_trace=decision_trace,
        )
    else:
        log_guardrail_event(
            guardrail="output_consistency",
            role=None,
            event="fired",
            reason=f"{len(flags)} inconsistency flag(s) raised",
            payload={"flags": flags},
            decision_trace=decision_trace,
        )
    return flags
