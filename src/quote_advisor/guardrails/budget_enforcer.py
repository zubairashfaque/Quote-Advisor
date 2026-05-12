"""Token-budget enforcement for LLM calls.

Pre-flight: count input tokens via tiktoken; raise if > max_input_tokens.
Post-flight: read AIMessage usage metadata; raise if total_tokens > max_total_tokens.

Output-token enforcement is delegated to the provider via the ``max_tokens``
argument passed to ``init_chat_model``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..budget_registry import AgentBudget

_LOG = logging.getLogger(__name__)
_tiktoken_warned = False
_no_usage_warned: set[str] = set()


class BudgetExceededError(Exception):
    """Raised when an LLM call exceeds the per-role budget."""

    def __init__(self, *, role: str, dimension: str, actual: int, limit: int) -> None:
        self.role = role
        self.dimension = dimension
        self.actual = actual
        self.limit = limit
        super().__init__(
            f"role={role} dimension={dimension} actual={actual} > limit={limit}"
        )


def _resolve_tiktoken_encoding(model: str):
    """Return a tiktoken Encoding for the model, or None on miss."""
    try:
        import tiktoken
    except ImportError:
        global _tiktoken_warned
        if not _tiktoken_warned:
            _LOG.warning("tiktoken not installed; falling back to char/4 token heuristic.")
            _tiktoken_warned = True
        return None

    bare = model.split(":", 1)[-1]
    try:
        return tiktoken.encoding_for_model(bare)
    except KeyError:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def count_input_tokens(messages: list[dict[str, Any]], *, model: str) -> int:
    """Count tokens across a list of role/content message dicts.

    Falls back to ``len(serialized) // 4`` when tiktoken is unavailable.
    """
    encoding = _resolve_tiktoken_encoding(model)
    if encoding is None:
        serialized = json.dumps(messages, default=str, ensure_ascii=False)
        return max(1, len(serialized) // 4)

    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(encoding.encode(content))
        elif isinstance(content, list):
            for chunk in content:
                if isinstance(chunk, dict):
                    text = chunk.get("text") or chunk.get("content") or ""
                    if isinstance(text, str):
                        total += len(encoding.encode(text))
        total += 4   # per-message overhead
    return total


def check_pre_flight(
    *,
    role: str,
    messages: list[dict[str, Any]],
    budget: AgentBudget,
    model: str,
) -> None:
    """Raise BudgetExceededError if input tokens exceed budget."""
    actual = count_input_tokens(messages, model=model)
    if actual > budget.max_input_tokens:
        raise BudgetExceededError(
            role=role,
            dimension="input_tokens",
            actual=actual,
            limit=budget.max_input_tokens,
        )


def _extract_total_tokens(ai_message: Any) -> int | None:
    """Pull total_tokens out of LangChain's two possible locations."""
    usage_meta = getattr(ai_message, "usage_metadata", None)
    if isinstance(usage_meta, dict):
        total = usage_meta.get("total_tokens")
        if isinstance(total, (int, float)) and not isinstance(total, bool):
            return int(total)
    resp_meta = getattr(ai_message, "response_metadata", None) or {}
    token_usage = resp_meta.get("token_usage") if isinstance(resp_meta, dict) else None
    if isinstance(token_usage, dict):
        total = token_usage.get("total_tokens")
        if isinstance(total, (int, float)) and not isinstance(total, bool):
            return int(total)
    return None


def check_post_flight(*, role: str, ai_message: Any, budget: AgentBudget) -> None:
    """Raise BudgetExceededError if total tokens exceed budget."""
    total = _extract_total_tokens(ai_message)
    if total is None:
        if role not in _no_usage_warned:
            _LOG.warning(
                "role=%s: response carries no token usage metadata; "
                "post-flight budget check skipped.",
                role,
            )
            _no_usage_warned.add(role)
        return
    if total > budget.max_total_tokens:
        raise BudgetExceededError(
            role=role,
            dimension="total_tokens",
            actual=total,
            limit=budget.max_total_tokens,
        )
