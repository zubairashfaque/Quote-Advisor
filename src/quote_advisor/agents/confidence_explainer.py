"""Confidence Explainer (DEC-0012, deferred to v1.1 — see README §17).

DEC-0004 (v1.0): the deterministic 4-signal aggregator computes the number,
the per-dimension breakdown, the hard cap (0.5 if statutory non-compliance), and the
[0.05, 0.95] clamp. None of that involves an LLM.

This module adds ONE cheap LLM call afterwards whose only job is to write a 2-3
sentence ``rationale_summary`` paragraph for the user. The LLM has no ability to
modify any number; it only writes prose. On any failure the paragraph is None and
the pipeline continues with the deterministic number unchanged.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .. import prompts
from ..llm_registry import AgentRole, get_llm
from ..schemas import ConfidenceBreakdown


class _ExplainerEmission(BaseModel):
    """LLM emits this; we extract the string."""

    rationale_summary: str = Field(..., min_length=20, max_length=600)


def explain(
    breakdown: ConfidenceBreakdown,
    state: dict[str, Any],
) -> str | None:
    """Return a 2-3 sentence paragraph or None on any failure.

    Inputs:
        breakdown : the deterministic ConfidenceBreakdown from compute_confidence()
        state     : the GraphState — used to extract top trace drivers

    The LLM never sees writeable references; it only gets a read-only summary
    of the breakdown plus the top 3-5 pricing/risk drivers from the DecisionTrace.
    """
    try:
        llm = get_llm(AgentRole.CONFIDENCE_EXPLAINER).with_structured_output(_ExplainerEmission)
        user_payload = _build_user_payload(breakdown, state)
        result: _ExplainerEmission = llm.invoke(
            [
                {"role": "system", "content": prompts.CONFIDENCE_EXPLAINER},
                {"role": "user",   "content": user_payload},
            ]
        )
        return result.rationale_summary.strip()
    except Exception:  # noqa: BLE001
        # Pipeline never fails because of explainer.
        return None


# =============================================================================
# Internals
# =============================================================================


def _build_user_payload(breakdown: ConfidenceBreakdown, state: dict[str, Any]) -> str:
    """Assemble the read-only context string the explainer LLM sees."""
    field_treatments = state.get("field_treatments") or {}
    triggered_rules = state.get("triggered_rules") or []
    council = breakdown.council_verdict or "not_invoked"
    has_neutral_credit = field_treatments.get("credit_score", "").startswith("neutral")
    factor_chain = state.get("factor_chain") or []

    drivers = _top_drivers(factor_chain)

    lines: list[str] = [
        "DETERMINISTIC CONFIDENCE BREAKDOWN (do not restate numbers; do not contradict):",
        f"  overall:    {breakdown.overall:.3f}",
        f"  risk:       {breakdown.risk if breakdown.risk is not None else 'n/a'}",
        f"  coverage:   {breakdown.coverage if breakdown.coverage is not None else 'n/a'}",
        f"  pricing:    {breakdown.pricing if breakdown.pricing is not None else 'n/a'}",
        f"  grounding:  {breakdown.grounding if breakdown.grounding is not None else 'n/a'}",
        f"  council:    invoked={breakdown.council_invoked}, verdict={council}",
        "",
        f"Statutory rules fired:  {len(triggered_rules)}",
        f"Statutory violations:   {len(state.get('statutory_violations') or [])}",
        f"FL no-credit treatment: {'YES' if has_neutral_credit else 'no'}",
        "",
        "Top pricing drivers (multiplier × name):",
        *(f"  - {d}" for d in drivers),
    ]
    return "\n".join(lines)


def _top_drivers(factor_chain: list[Any], k: int = 5) -> list[str]:
    """Pick the k strongest multipliers (furthest from 1.0)."""
    rows: list[tuple[float, str]] = []
    for entry in factor_chain:
        if hasattr(entry, "model_dump"):
            d = entry.model_dump()
        else:
            d = dict(entry)
        m = d.get("multiplier")
        n = d.get("name") or d.get("step_id") or "?"
        if isinstance(m, (int, float)):
            rows.append((abs(float(m) - 1.0), f"{m:.3f}× {n}"))
    rows.sort(reverse=True)
    return [s for _, s in rows[:k]] or ["(no factor chain available)"]


__all__ = ["explain"]
