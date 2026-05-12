"""Numeric range clamp for premium output.

Premium estimates outside [floor, ceiling] per state are clamped and a
warning is appended. Floors / ceilings are conservative defensible values
- not regulatory minimums.

Emits one audit_logger event per call (passed | fired).
"""

from __future__ import annotations

from ..schemas import DecisionNode, PremiumRange
from .audit_logger import log_guardrail_event

_PER_STATE_FLOOR_USD = {
    "CA":      400.0,
    "FL":      900.0,
    "DEFAULT": 300.0,
}

_PER_STATE_CEILING_USD = {
    "CA":      75_000.0,
    "FL":      60_000.0,
    "DEFAULT": 50_000.0,
}


def clamp_premium_range(
    pr: PremiumRange | None,
    state: str | None,
    *,
    decision_trace: list[DecisionNode] | None = None,
) -> tuple[PremiumRange | None, list[str]]:
    """Return (clamped PremiumRange, warnings) - warnings is empty when no clamping fired."""
    if pr is None:
        return pr, []
    state_code = (state or "").upper()
    floor = _PER_STATE_FLOOR_USD.get(state_code, _PER_STATE_FLOOR_USD["DEFAULT"])
    ceiling = _PER_STATE_CEILING_USD.get(state_code, _PER_STATE_CEILING_USD["DEFAULT"])

    warnings: list[str] = []
    low, high = pr.low, pr.high
    if low < floor:
        warnings.append(f"premium_range.low ${low:,.0f} clamped to state floor ${floor:,.0f}")
        low = floor
    if high > ceiling:
        warnings.append(f"premium_range.high ${high:,.0f} clamped to state ceiling ${ceiling:,.0f}")
        high = ceiling
    if high < low:
        high = low

    if not warnings:
        log_guardrail_event(
            guardrail="range_clamp",
            role=None,
            event="passed",
            reason=f"premium_range in [{floor:.0f}, {ceiling:.0f}] for state={state_code or 'DEFAULT'}",
            payload={"low": pr.low, "high": pr.high, "state": state_code},
            decision_trace=decision_trace,
        )
        return pr, []

    clamped = PremiumRange(low=round(low, 2), high=round(high, 2), currency=pr.currency)
    log_guardrail_event(
        guardrail="range_clamp",
        role=None,
        event="fired",
        reason="; ".join(warnings),
        payload={
            "state":        state_code,
            "original_low": pr.low,
            "original_high": pr.high,
            "clamped_low":  clamped.low,
            "clamped_high": clamped.high,
        },
        decision_trace=decision_trace,
    )
    return clamped, warnings
