"""PII scrubber for logging boundary.

Hashes / redacts identifiers before they hit stdout / structured logs.
Emits one audit_logger event per call (fired if any key redacted, passed otherwise).
Demo profiles in ``data/profiles/`` do not contain name / address / DOB, but
the scrubber ships ready for real intake forms.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from ..schemas import DecisionNode
from .audit_logger import log_guardrail_event

_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE_RE = re.compile(r"\b(?:\+1[-. ]?)?(?:\(\d{3}\)|\d{3})[-. ]?\d{3}[-. ]?\d{4}\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")

_REDACTED_KEYS = {
    "name",
    "first_name",
    "last_name",
    "address",
    "street",
    "ssn",
    "dob",
    "date_of_birth",
    "phone",
    "email",
    "policy_number",
}


def _hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()[:12]


def _scrub_recurse(payload: Any, redacted_count: list[int]) -> Any:
    if isinstance(payload, dict):
        out = {}
        for k, v in payload.items():
            if k.lower() in _REDACTED_KEYS:
                out[k] = "<redacted>"
                redacted_count[0] += 1
            else:
                out[k] = _scrub_recurse(v, redacted_count)
        return out
    if isinstance(payload, list):
        return [_scrub_recurse(v, redacted_count) for v in payload]
    if isinstance(payload, str):
        s = _SSN_RE.sub("<ssn>", payload)
        s = _CREDIT_CARD_RE.sub("<cc>", s)
        s = _PHONE_RE.sub("<phone>", s)
        s = _EMAIL_RE.sub("<email>", s)
        return s
    return payload


def scrub_for_logging(
    payload: Any,
    *,
    decision_trace: list[DecisionNode] | None = None,
) -> Any:
    """Return a deep-copy of ``payload`` with PII fields redacted and stringy
    payloads scrubbed of common PII regexes.

    Emits one audit_logger event: ``fired`` if any key was redacted, ``passed``
    otherwise.
    """
    redacted = [0]
    out = _scrub_recurse(payload, redacted)
    if redacted[0] > 0:
        log_guardrail_event(
            guardrail="pii_scrubber",
            role=None,
            event="fired",
            reason=f"redacted {redacted[0]} key(s)",
            payload={"redacted_count": redacted[0]},
            decision_trace=decision_trace,
        )
    else:
        log_guardrail_event(
            guardrail="pii_scrubber",
            role=None,
            event="passed",
            reason="no PII keys found",
            decision_trace=decision_trace,
        )
    return out
