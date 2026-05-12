"""Statutory gate guardrail wrapper.

Forwards to ``statutory_rules_engine.apply``. Kept as a separate module so the
graph-side and CLI-side wrappings stay isolated from the engine internals.

Emits one audit_logger event per call.
"""

from __future__ import annotations

from typing import Any

from .. import statutory_rules_engine
from ..schemas import DecisionNode, StatutoryEngineOutput
from .audit_logger import log_guardrail_event


def apply_statutory_gate(
    raw_profile: dict[str, Any],
    *,
    context: dict[str, Any] | None = None,
    decision_trace: list[DecisionNode] | None = None,
) -> StatutoryEngineOutput:
    """Apply the StatutoryRulesEngine. Convenience facade for non-graph callers."""
    out = statutory_rules_engine.apply(raw_profile, context=context)
    rules = getattr(out, "triggered_rules", None) or []
    violations = getattr(out, "statutory_violations", None) or []
    if violations:
        event, reason = "fired", f"{len(violations)} violation(s)"
    else:
        event, reason = "passed", f"{len(rules)} rule(s) triggered, no violations"
    log_guardrail_event(
        guardrail="statutory_gate",
        role=None,
        event=event,
        reason=reason,
        payload={"rules_count": len(rules), "violations_count": len(violations)},
        decision_trace=decision_trace,
    )
    return out
