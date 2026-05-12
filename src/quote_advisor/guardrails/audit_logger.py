"""Structured guardrail audit logger.

A single surface every guardrail emits through. Each call emits one plain-text
line to stderr AND (optionally) appends a make_node()-built DecisionNode to the
provided decision_trace list.

This is additive: it does NOT replace the customer-facing ``warnings: list[str]``
on QuoteOutput.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Literal

from ..decision_trace import make_node
from ..schemas import DecisionNode

GuardrailEvent = Literal["fired", "passed", "fallback", "abort"]


def log_guardrail_event(
    *,
    guardrail: str,
    role: str | None,
    event: GuardrailEvent,
    reason: str,
    payload: dict[str, Any] | None = None,
    decision_trace: list[DecisionNode] | None = None,
) -> None:
    """Emit a single structured guardrail event.

    1. Always writes one plain line to stderr.
    2. If ``decision_trace`` is provided, appends a make_node() entry in-place.
    """
    payload_dict = dict(payload or {})
    role_str = role or "-"
    payload_json = json.dumps(payload_dict, separators=(",", ":"), default=str)
    line = (
        f"[GUARDRAIL {guardrail}] role={role_str} event={event} "
        f"reason={reason} payload={payload_json}"
    )
    print(line, file=sys.stderr, flush=True)

    if decision_trace is not None:
        node_payload = {
            **payload_dict,
            "guardrail": guardrail,
            "role": role,
            "event": event,
            "reason": reason,
        }
        decision_trace.append(
            make_node(
                agent="GuardrailAudit",
                summary=f"[{guardrail}] {event}: {reason}",
                payload=node_payload,
            )
        )
