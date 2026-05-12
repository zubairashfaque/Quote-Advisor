"""Retry-with-validation guardrail.

If a structured-output call fails Pydantic validation, retry up to ``max_retries``
times, each time appending the validation errors to the prompt so the model
can self-correct. Emits one audit_logger event per attempt.
"""

from __future__ import annotations

import json
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError

from ..schemas import DecisionNode
from .audit_logger import log_guardrail_event

T = TypeVar("T", bound=BaseModel)


def retry_with_validation(
    invoker: Callable[[list[dict[str, Any]]], Any],
    schema: type[T],
    base_messages: list[dict[str, Any]],
    *,
    max_retries: int = 2,
    decision_trace: list[DecisionNode] | None = None,
    role: str | None = None,
) -> T | None:
    """Call ``invoker(messages)`` and validate the result against ``schema``.

    On validation failure, retry up to ``max_retries`` times with the validation
    error appended to the message list. Emits one audit_logger event per attempt
    (``fired`` on failure, ``passed`` on success, ``fallback`` on exhaustion).
    Returns the validated model on success or ``None`` after exhaustion (caller
    should fall back to a conservative default).
    """
    messages = list(base_messages)
    last_errors: Any = None
    for attempt in range(max_retries + 1):
        result = invoker(messages)
        try:
            payload = result if isinstance(result, dict) else getattr(result, "content", str(result))
            if isinstance(payload, str):
                payload = json.loads(payload)
            validated = schema.model_validate(payload)
            log_guardrail_event(
                guardrail="retry_validator",
                role=role,
                event="passed",
                reason=f"{schema.__name__} validated on attempt {attempt + 1}",
                payload={"attempt": attempt + 1, "schema": schema.__name__},
                decision_trace=decision_trace,
            )
            return validated
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            last_errors = exc.errors() if isinstance(exc, ValidationError) else str(exc)
            log_guardrail_event(
                guardrail="retry_validator",
                role=role,
                event="fired",
                reason=f"{schema.__name__} validation failed on attempt {attempt + 1}",
                payload={
                    "attempt": attempt + 1,
                    "schema": schema.__name__,
                    "errors": last_errors if isinstance(last_errors, (str, list)) else str(last_errors),
                },
                decision_trace=decision_trace,
            )
            if attempt == max_retries:
                log_guardrail_event(
                    guardrail="retry_validator",
                    role=role,
                    event="fallback",
                    reason=f"exhausted {max_retries + 1} attempts; returning None",
                    payload={"schema": schema.__name__},
                    decision_trace=decision_trace,
                )
                break
            messages = list(base_messages) + [
                {
                    "role": "user",
                    "content": (
                        f"Your previous output failed validation against schema {schema.__name__}: "
                        f"{last_errors}. Re-emit the structured output correctly."
                    ),
                }
            ]
    return None
