"""Guardrails - five wired layers (V4 Part 7).

  1. input_validation     - Pydantic on CustomerProfile + Profile B branching
  2. output_consistency   - cross-agent consistency checks (also in validator_agent)
  3. retry_validator      - retry-with-validation, max 2 retries
  4. statutory_gate       - wrapper around StatutoryRulesEngine for graph use
  5. fallback handling    - distributed (each agent has a fallback path)

Plus advisory layers (always wired): ``pii_scrubber``, ``prompt_injection_sanitizer``,
``range_clamp``.
"""

from .audit_logger import log_guardrail_event
from .budget_enforcer import BudgetExceededError, check_post_flight, check_pre_flight
from .input_validation import sanitise_input, validate_input
from .output_consistency import check_output
from .pii_scrubber import scrub_for_logging
from .prompt_injection_sanitizer import sanitise_text_field
from .range_clamp import clamp_premium_range
from .retry_validator import retry_with_validation
from .statutory_gate import apply_statutory_gate

__all__ = [
    "validate_input",
    "sanitise_input",
    "check_output",
    "retry_with_validation",
    "apply_statutory_gate",
    "scrub_for_logging",
    "sanitise_text_field",
    "clamp_premium_range",
    "log_guardrail_event",
    "BudgetExceededError",
    "check_pre_flight",
    "check_post_flight",
]
