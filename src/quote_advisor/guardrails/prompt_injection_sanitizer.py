"""Prompt-injection sanitiser for free-text profile fields.

Strips role-play markers, system-prompt sequences, and known jailbreak prefixes.
Applied to every string-valued field BEFORE Pydantic validation (see
``input_validation.sanitise_input``).

Emits one audit_logger event per call (passed | fired).
"""

from __future__ import annotations

import re

from ..schemas import DecisionNode
from .audit_logger import log_guardrail_event

_INJECTION_PATTERNS = [
    re.compile(r"\[INST\]", re.IGNORECASE),
    re.compile(r"\[/INST\]", re.IGNORECASE),
    re.compile(r"<\|im_start\|>", re.IGNORECASE),
    re.compile(r"<\|im_end\|>", re.IGNORECASE),
    re.compile(r"<\|system\|>", re.IGNORECASE),
    re.compile(r"<\|user\|>", re.IGNORECASE),
    re.compile(r"<\|assistant\|>", re.IGNORECASE),
    re.compile(r"###\s*(?:System|Assistant|User|Instruction)\s*:", re.IGNORECASE),
    re.compile(r"^\s*(?:ignore|disregard|override)\s+(?:previous|prior|all|the\s+above)\s+instructions", re.IGNORECASE | re.MULTILINE),
]


def sanitise_text_field(
    value: str,
    *,
    decision_trace: list[DecisionNode] | None = None,
) -> str:
    """Strip injection markers; return a sanitised copy. Idempotent."""
    s = value
    triggered: list[str] = []
    for pat in _INJECTION_PATTERNS:
        if pat.search(s):
            triggered.append(pat.pattern)
            s = pat.sub("", s)
    cleaned = s.strip()

    if triggered:
        log_guardrail_event(
            guardrail="prompt_injection_sanitizer",
            role=None,
            event="fired",
            reason=f"matched {len(triggered)} injection pattern(s)",
            payload={"patterns": triggered},
            decision_trace=decision_trace,
        )
    else:
        log_guardrail_event(
            guardrail="prompt_injection_sanitizer",
            role=None,
            event="passed",
            reason="no injection patterns matched",
            decision_trace=decision_trace,
        )
    return cleaned
