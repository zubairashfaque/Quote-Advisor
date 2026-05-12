"""StateDiffTool - load-bearing diff for the Counterfactual agent.

Compares two GraphState dicts (or any flat-ish dicts) and returns the changed
keys with before/after values. Used to render the 'what changed when you
removed the pool' diff table.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class StateDiffInput(BaseModel):
    state_a: dict[str, Any] = Field(..., description="Base state (before mutation)")
    state_b: dict[str, Any] = Field(..., description="Counterfactual state (after mutation)")
    keys_of_interest: list[str] | None = Field(
        default=None,
        description="Optional whitelist; if provided, only these keys are diffed",
    )


class FieldChange(BaseModel):
    key: str
    before: Any
    after: Any


class StateDiffOutput(BaseModel):
    changed_fields: list[FieldChange]
    only_in_a: list[str]
    only_in_b: list[str]


@tool("state_diff", args_schema=StateDiffInput)
def state_diff(
    state_a: dict[str, Any],
    state_b: dict[str, Any],
    keys_of_interest: list[str] | None = None,
) -> dict:
    """Diff two state dicts; return the keys whose values differ plus keys present in only one side.

    Used by the Counterfactual agent to summarise 'drivers_changed' (e.g., pool surcharge multiplier 1.12 -> 1.0, liability_factor 'high' -> 'medium').
    """
    a_keys = set(state_a.keys())
    b_keys = set(state_b.keys())
    common = a_keys & b_keys
    if keys_of_interest:
        common = common & set(keys_of_interest)

    changed: list[FieldChange] = []
    for k in sorted(common):
        if state_a.get(k) != state_b.get(k):
            changed.append(FieldChange(key=k, before=state_a.get(k), after=state_b.get(k)))

    only_a = sorted(a_keys - b_keys) if not keys_of_interest else []
    only_b = sorted(b_keys - a_keys) if not keys_of_interest else []
    return StateDiffOutput(changed_fields=changed, only_in_a=only_a, only_in_b=only_b).model_dump()
