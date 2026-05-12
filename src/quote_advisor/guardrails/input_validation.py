"""Input validation guardrail.

Wraps the Pydantic ``CustomerProfile`` validator with the graceful-handling
contract: Profile B (``credit_score=null``) is allowed and flagged for the
Pricing path's neutral-treatment, not rejected.

Every call emits one audit_logger event (passed | fired).
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from ..schemas import CustomerProfile, DecisionNode
from .audit_logger import log_guardrail_event
from .prompt_injection_sanitizer import sanitise_text_field


class InputValidationError(Exception):
    """Hard rejection - profile fails Pydantic validation."""


def validate_input(
    raw: dict[str, Any],
    *,
    decision_trace: list[DecisionNode] | None = None,
) -> tuple[CustomerProfile, list[str]]:
    """Return (validated CustomerProfile, list of warnings).

    Raises ``InputValidationError`` on hard failure. Emits one audit_logger event.
    """
    if not isinstance(raw, dict):
        log_guardrail_event(
            guardrail="input_validation",
            role=None,
            event="fired",
            reason=f"profile not a dict; got {type(raw).__name__}",
            decision_trace=decision_trace,
        )
        raise InputValidationError(f"Profile must be a JSON object; got {type(raw).__name__}")
    try:
        profile = CustomerProfile.model_validate(raw)
    except ValidationError as ve:
        log_guardrail_event(
            guardrail="input_validation",
            role=None,
            event="fired",
            reason="pydantic validation failed",
            payload={"errors": ve.errors()},
            decision_trace=decision_trace,
        )
        raise InputValidationError(f"Pydantic validation failed: {ve.errors()}") from ve

    warnings: list[str] = []
    if profile.credit_score is None:
        warnings.append("credit_score=null: handled per CA Prop 103 (drop) or FL §626.9741 (neutral 1.0×)")
        log_guardrail_event(
            guardrail="input_validation",
            role=None,
            event="fired",
            reason="credit_score is null; statutory neutral/drop path will apply",
            payload={"field": "credit_score"},
            decision_trace=decision_trace,
        )

    log_guardrail_event(
        guardrail="input_validation",
        role=None,
        event="passed",
        reason="profile validated",
        payload={"warnings_count": len(warnings)},
        decision_trace=decision_trace,
    )
    return profile, warnings


def sanitise_input(raw: dict[str, Any]) -> dict[str, Any]:
    """Apply prompt-injection sanitisation to all string fields before validation."""
    if not isinstance(raw, dict):
        return raw
    out: dict[str, Any] = {}
    for k, v in raw.items():
        out[k] = sanitise_text_field(v) if isinstance(v, str) else v
    return out
