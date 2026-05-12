"""Per-agent LLM registry (DEC-0008).

Every agent and Council persona resolves its LLM via ``get_llm(role)``.
Resolution order (highest precedence wins):

    1. Environment variable  ``QA_LLM_<ROLE>``           (e.g., ``QA_LLM_RISK_AGENT=openai:gpt-4o``)
    2. Local YAML override   ``configs/llm_roles.local.yaml``
    3. Committed YAML        ``configs/llm_roles.yaml``
    4. Hardcoded defaults    ``DEFAULTS`` (in this module)

All values are strings of the form ``"<provider>:<model>"`` (e.g.,
``"anthropic:claude-sonnet-4-6"``); resolution uses LangChain's
``init_chat_model`` so adding a provider is one config edit.

No agent code references a provider SDK directly.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from .configuration import REPO_ROOT, get_settings

if TYPE_CHECKING:  # avoid heavy import at module load
    from langchain_core.language_models.chat_models import BaseChatModel


# =============================================================================
# Role enum - 14 distinct LLM seats
# =============================================================================


class AgentRole(str, Enum):
    INTENT_CLASSIFIER     = "INTENT_CLASSIFIER"
    ELIGIBILITY_TRIAGE    = "ELIGIBILITY_TRIAGE"
    RISK_AGENT            = "RISK_AGENT"
    COVERAGE_PLANNER      = "COVERAGE_PLANNER"
    COVERAGE_EXECUTOR     = "COVERAGE_EXECUTOR"
    PRICING_PLANNER       = "PRICING_PLANNER"
    PRICING_SOLVER        = "PRICING_SOLVER"
    VALIDATOR             = "VALIDATOR"
    COUNCIL_UNDERWRITER   = "COUNCIL_UNDERWRITER"
    COUNCIL_ADVOCATE      = "COUNCIL_ADVOCATE"
    COUNCIL_ACTUARY       = "COUNCIL_ACTUARY"
    COUNCIL_COMPLIANCE    = "COUNCIL_COMPLIANCE"
    COUNTERFACTUAL        = "COUNTERFACTUAL"
    FOLLOWUP_EXPLAIN      = "FOLLOWUP_EXPLAIN"
    STATUTORY_AGENT       = "STATUTORY_AGENT"
    CONFIDENCE_EXPLAINER  = "CONFIDENCE_EXPLAINER"


# =============================================================================
# Hardcoded defaults (lowest precedence)
# =============================================================================


DEFAULTS: dict[AgentRole, str] = {
    AgentRole.INTENT_CLASSIFIER:     "openai:gpt-4o-mini",
    AgentRole.ELIGIBILITY_TRIAGE:    "openai:gpt-4o-mini",
    AgentRole.RISK_AGENT:            "openai:gpt-4o",
    AgentRole.COVERAGE_PLANNER:      "openai:gpt-4o",
    AgentRole.COVERAGE_EXECUTOR:     "openai:gpt-4o-mini",
    AgentRole.PRICING_PLANNER:       "openai:gpt-4o",
    AgentRole.PRICING_SOLVER:        "openai:gpt-4o",
    AgentRole.VALIDATOR:             "openai:gpt-4o",
    AgentRole.COUNCIL_UNDERWRITER:   "openai:gpt-4o",
    AgentRole.COUNCIL_ADVOCATE:      "openai:gpt-4o",
    AgentRole.COUNCIL_ACTUARY:       "openai:gpt-4o",
    AgentRole.COUNCIL_COMPLIANCE:    "openai:gpt-4o",  # never downgrade - VETO power
    AgentRole.COUNTERFACTUAL:        "openai:gpt-4o",
    AgentRole.FOLLOWUP_EXPLAIN:      "openai:gpt-4o-mini",
    AgentRole.STATUTORY_AGENT:       "openai:gpt-4o",          # ReAct over RAG; full-tier
    AgentRole.CONFIDENCE_EXPLAINER:  "openai:gpt-4o-mini",     # prose only; cheap
}


# =============================================================================
# YAML loaders
# =============================================================================


CONFIGS_DIR = REPO_ROOT / "configs"


def _load_yaml(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return {str(k).upper(): str(v) for k, v in data.items() if v is not None}


@lru_cache(maxsize=1)
def _resolved_role_table() -> dict[AgentRole, str]:
    """Resolve every role once; cache the merged table."""
    table: dict[AgentRole, str] = dict(DEFAULTS)

    yaml_main = _load_yaml(CONFIGS_DIR / "llm_roles.yaml")
    yaml_local = _load_yaml(CONFIGS_DIR / "llm_roles.local.yaml")

    for role in AgentRole:
        if role.value in yaml_main:
            table[role] = yaml_main[role.value]
        if role.value in yaml_local:
            table[role] = yaml_local[role.value]
        env_key = f"QA_LLM_{role.value}"
        env_val = os.environ.get(env_key)
        if env_val:
            table[role] = env_val
    return table


# =============================================================================
# Public API
# =============================================================================


def resolve_model_id(role: AgentRole | str) -> str:
    """Return the resolved ``provider:model`` identifier for a role."""
    if isinstance(role, str):
        role = AgentRole(role)
    return _resolved_role_table()[role]


def all_resolved() -> dict[str, str]:
    """For ``--llm-trace``: print every role -> model resolution at startup."""
    table = _resolved_role_table()
    return {role.value: model for role, model in sorted(table.items(), key=lambda kv: kv[0].value)}


class BudgetedChatModel:
    """Thin proxy around a BaseChatModel that enforces per-role token budgets.

    Pre-flight: counts input tokens, raises BudgetExceededError if over.
    Post-flight: reads usage metadata, raises if total exceeds budget.
    All other attributes / methods delegate to the wrapped model.
    """

    def __init__(self, inner: "BaseChatModel", role: str, model_id: str, budget: Any) -> None:
        self._inner = inner
        self._qa_role = role
        self._qa_model_id = model_id
        self._qa_budget = budget

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _normalize_messages(self, input_: Any) -> list[dict[str, Any]]:
        if isinstance(input_, str):
            return [{"role": "user", "content": input_}]
        if isinstance(input_, list):
            norm: list[dict[str, Any]] = []
            for m in input_:
                if isinstance(m, dict):
                    norm.append(m)
                else:
                    role = getattr(m, "type", "user")
                    content = getattr(m, "content", str(m))
                    norm.append({"role": role, "content": content})
            return norm
        return [{"role": "user", "content": str(input_)}]

    def invoke(self, input_: Any, *args: Any, **kwargs: Any) -> Any:
        from .guardrails.audit_logger import log_guardrail_event
        from .guardrails.budget_enforcer import (
            BudgetExceededError,
            check_post_flight,
            check_pre_flight,
        )

        # Pop trace BEFORE any branch so it never leaks into the inner provider's kwargs.
        decision_trace = kwargs.pop("_qa_decision_trace", None)

        messages = self._normalize_messages(input_)
        try:
            check_pre_flight(
                role=self._qa_role,
                messages=messages,
                budget=self._qa_budget,
                model=self._qa_model_id,
            )
        except BudgetExceededError as exc:
            log_guardrail_event(
                guardrail="token_budget",
                role=self._qa_role,
                event=self._qa_budget.action_on_breach.value,
                reason=str(exc),
                payload={"dimension": exc.dimension, "actual": exc.actual, "limit": exc.limit},
                decision_trace=decision_trace,
            )
            if self._qa_budget.action_on_breach.value != "warn":
                raise

        result = self._inner.invoke(input_, *args, **kwargs)

        try:
            check_post_flight(role=self._qa_role, ai_message=result, budget=self._qa_budget)
        except BudgetExceededError as exc:
            log_guardrail_event(
                guardrail="token_budget",
                role=self._qa_role,
                event=self._qa_budget.action_on_breach.value,
                reason=str(exc),
                payload={"dimension": exc.dimension, "actual": exc.actual, "limit": exc.limit},
                decision_trace=decision_trace,
            )
            if self._qa_budget.action_on_breach.value == "warn":
                return result
            raise
        return result


def get_llm(role: AgentRole | str, *, temperature: float = 0.0, **kwargs: Any) -> "BaseChatModel":
    """Return an instantiated chat model for a role, wrapped with budget enforcement.

    The returned object is a BudgetedChatModel proxy that pre/post-flight checks
    input/output tokens against ``configs/agent_budgets.yaml``; on breach it routes
    to the role's configured action (fallback | abort | warn) and logs via
    audit_logger.
    """
    # Ensure LangSmith env vars are populated before init_chat_model constructs the client.
    get_settings()

    from langchain.chat_models import init_chat_model  # imported lazily

    from .budget_registry import get_budget

    if isinstance(role, str):
        role_enum = AgentRole(role)
    else:
        role_enum = role

    model_id = resolve_model_id(role_enum)
    budget = get_budget(role_enum)

    # Cap output via the provider's own max_tokens; this is free enforcement.
    kwargs.setdefault("max_tokens", budget.max_output_tokens)

    inner = init_chat_model(model_id, temperature=temperature, **kwargs)
    return BudgetedChatModel(inner, role=role_enum.value, model_id=model_id, budget=budget)


def reset_cache() -> None:
    """Test/dev hook: clear the resolved role table cache (e.g., after env mutation)."""
    _resolved_role_table.cache_clear()
