"""Per-agent budget registry.

Mirrors ``llm_registry`` for safety limits. Resolution order:
    1. Environment: QA_BUDGET_<ROLE>_<FIELD>=<value>
    2. configs/agent_budgets.yaml role override
    3. configs/agent_budgets.yaml defaults block
    4. BAKED_DEFAULTS in this module (lowest precedence; survives missing yaml)
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from functools import lru_cache
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from .configuration import REPO_ROOT
from .llm_registry import AgentRole

_LOG = logging.getLogger(__name__)

_BUDGETS_YAML = REPO_ROOT / "configs" / "agent_budgets.yaml"

_BUDGET_FIELDS = (
    "max_react_iterations",
    "max_input_tokens",
    "max_output_tokens",
    "max_total_tokens",
    "action_on_breach",
)


class BudgetAction(str, Enum):
    FALLBACK = "fallback"
    ABORT    = "abort"
    WARN     = "warn"


class AgentBudget(BaseModel):
    """Per-role enforcement budget."""

    max_react_iterations: int = Field(ge=1, le=32)
    max_input_tokens:     int = Field(ge=100, le=128_000)
    max_output_tokens:    int = Field(ge=50,  le=16_000)
    max_total_tokens:     int = Field(ge=200, le=200_000)
    action_on_breach:     BudgetAction

    @field_validator("max_total_tokens")
    @classmethod
    def _total_covers_output(cls, v: int, info: Any) -> int:
        out = info.data.get("max_output_tokens")
        if out is not None and v < out:
            raise ValueError(f"max_total_tokens ({v}) must be >= max_output_tokens ({out})")
        return v


BAKED_DEFAULTS: dict[str, Any] = {
    "max_react_iterations": 8,
    "max_input_tokens":     8000,
    "max_output_tokens":    2000,
    "max_total_tokens":     10_000,
    "action_on_breach":     "fallback",
}

_warned_unknown_roles: set[str] = set()


def _load_yaml() -> dict[str, Any]:
    if not _BUDGETS_YAML.exists():
        _LOG.warning("agent_budgets.yaml missing at %s; using baked defaults only.", _BUDGETS_YAML)
        return {"defaults": dict(BAKED_DEFAULTS), "roles": {}}
    with _BUDGETS_YAML.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return {
        "defaults": {**BAKED_DEFAULTS, **(data.get("defaults") or {})},
        "roles":    data.get("roles") or {},
    }


def _apply_env_overrides(role_key: str, values: dict[str, Any]) -> dict[str, Any]:
    out = dict(values)
    for field in _BUDGET_FIELDS:
        env_key = f"QA_BUDGET_{role_key}_{field.upper()}"
        env_val = os.environ.get(env_key)
        if env_val is None:
            continue
        if field == "action_on_breach":
            out[field] = env_val.lower()
        else:
            try:
                out[field] = int(env_val)
            except ValueError as exc:
                raise ValueError(f"{env_key}={env_val!r} is not an integer") from exc
    # Ensure max_total_tokens >= max_output_tokens after env overrides.
    # An env var may raise max_output_tokens above the YAML max_total_tokens;
    # auto-expand max_total_tokens rather than raise a confusing cross-field error.
    max_out = out.get("max_output_tokens", 0)
    max_total = out.get("max_total_tokens", 0)
    if max_out and max_total and max_out > max_total:
        _LOG.warning(
            "QA_BUDGET_%s_MAX_OUTPUT_TOKENS=%d exceeds max_total_tokens=%d; "
            "auto-bumping max_total_tokens to match.",
            role_key, max_out, max_total,
        )
        out["max_total_tokens"] = max_out
    return out


@lru_cache(maxsize=1)
def _resolved_budget_table() -> dict[str, AgentBudget]:
    yaml_data = _load_yaml()
    defaults = yaml_data["defaults"]
    role_overrides = yaml_data["roles"]

    resolved: dict[str, AgentBudget] = {}
    for role in AgentRole:
        merged = {**defaults, **(role_overrides.get(role.value) or {})}
        merged = _apply_env_overrides(role.value, merged)
        resolved[role.value] = AgentBudget.model_validate(merged)
    return resolved


def get_budget(role: AgentRole | str) -> AgentBudget:
    """Return the AgentBudget for ``role``.

    Unknown role strings fall back to the defaults block (with a one-time warning).
    """
    role_key = role.value if isinstance(role, AgentRole) else str(role).upper()
    table = _resolved_budget_table()
    if role_key in table:
        return table[role_key]

    if role_key not in _warned_unknown_roles:
        _LOG.warning("unknown agent role %r; using budget defaults.", role_key)
        _warned_unknown_roles.add(role_key)
    yaml_data = _load_yaml()
    merged = _apply_env_overrides(role_key, dict(yaml_data["defaults"]))
    return AgentBudget.model_validate(merged)


def all_resolved() -> dict[str, AgentBudget]:
    return dict(_resolved_budget_table())
