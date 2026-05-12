"""Eligibility Triage (Tree-of-Thoughts).

Generates 4 candidate market routes, scores each, prunes, picks the highest.
Mostly deterministic scoring + LLM narrative.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .. import prompts
from ..decision_trace import append_node, make_node
from ..llm_registry import AgentRole, get_llm
from ..schemas import EligibilityRoute


class EligibilityBranch(BaseModel):
    route: EligibilityRoute
    score: float = Field(..., ge=0.0, le=1.0)
    rationale: str
    pruned: bool = False


class EligibilityResult(BaseModel):
    chosen_route: EligibilityRoute
    rationale: str
    branches: list[EligibilityBranch]
    evidence_ids: list[str] = Field(default_factory=list)


def _deterministic_branches(
    state_code: str,
    fhsz_likely: str | None,
    hurricane_tier: str | None,
) -> list[EligibilityBranch]:
    branches: list[EligibilityBranch] = []
    if state_code == "CA":
        admitted_score = 0.7 if fhsz_likely in {"Low", "Moderate", None} else 0.20
        fair_dic_score = 0.85 if fhsz_likely in {"High", "Very High"} else 0.30
        branches.append(EligibilityBranch(route=EligibilityRoute.ADMITTED, score=admitted_score, rationale=f"CA admitted-market viability vs FHSZ tier {fhsz_likely}"))
        branches.append(EligibilityBranch(route=EligibilityRoute.FAIR_DIC, score=fair_dic_score, rationale=f"CA FAIR Plan + DIC viability vs FHSZ tier {fhsz_likely}"))
        branches.append(EligibilityBranch(route=EligibilityRoute.CITIZENS, score=0.00, rationale="Citizens is FL-only", pruned=True))
        branches.append(EligibilityBranch(route=EligibilityRoute.SURPLUS_LINES, score=0.50, rationale="E&S always available; cost penalty -0.30 implicit"))
    elif state_code == "FL":
        admitted_score = 0.75 if hurricane_tier in {"Low", "Moderate", None} else 0.55
        citizens_score = 0.65 if hurricane_tier in {"High", "Very High"} else 0.35
        branches.append(EligibilityBranch(route=EligibilityRoute.ADMITTED, score=admitted_score, rationale=f"FL admitted-market viability vs hurricane tier {hurricane_tier}"))
        branches.append(EligibilityBranch(route=EligibilityRoute.CITIZENS, score=citizens_score, rationale=f"FL Citizens viability vs hurricane tier {hurricane_tier}"))
        branches.append(EligibilityBranch(route=EligibilityRoute.FAIR_DIC, score=0.00, rationale="FAIR Plan is CA-only", pruned=True))
        branches.append(EligibilityBranch(route=EligibilityRoute.SURPLUS_LINES, score=0.45, rationale="E&S always available; cost penalty"))
    else:
        branches.append(EligibilityBranch(route=EligibilityRoute.INFORMATIONAL, score=1.0, rationale="State outside CA/FL coverage"))
    return branches


def eligibility_node(state: dict[str, Any]) -> dict[str, Any]:
    """Score 4 market-route branches, prune dominated, select best."""
    sanitized = state.get("sanitized_profile") or {}
    state_code = (sanitized.get("state") or "").upper()
    market_route_hints = state.get("market_route_hints", []) or []

    # The Risk agent has not run yet; we work from preview signals only.
    fhsz_likely_hint = "High" if any("fair_plan" in h or "fhsz_high" in h for h in market_route_hints) else None
    hurricane_tier_hint = "Very High" if state_code == "FL" else None

    branches = _deterministic_branches(state_code, fhsz_likely_hint, hurricane_tier_hint)
    surviving = [b for b in branches if b.score >= 0.20 and not b.pruned]
    chosen = max(surviving, key=lambda b: b.score) if surviving else branches[0]

    rationale = (
        f"Route '{chosen.route.value}' chosen with score {chosen.score:.2f}; "
        f"surviving alternatives: {[b.route.value for b in surviving if b.route != chosen.route]}."
    )

    # Optional LLM narrative pass for richer rationale (kept out of test cost when desired).
    if state.get("disable_eligibility_narrative", False) is False:
        try:
            llm = get_llm(AgentRole.ELIGIBILITY_TRIAGE).with_structured_output(EligibilityResult)
            llm_result: EligibilityResult = llm.invoke(
                [
                    {"role": "system", "content": prompts.ELIGIBILITY_TRIAGE},
                    {
                        "role": "user",
                        "content": (
                            f"State: {state_code}. "
                            f"FHSZ hint: {fhsz_likely_hint}. Hurricane hint: {hurricane_tier_hint}. "
                            f"Deterministic branches: {[b.model_dump() for b in branches]}. "
                            f"Pre-selected: {chosen.route.value} (score {chosen.score:.2f}). "
                            "Confirm or override only if there is statutory or rule-based reason; otherwise echo."
                        ),
                    },
                ]
            )
            chosen = next((b for b in branches if b.route == llm_result.chosen_route), chosen)
            rationale = llm_result.rationale or rationale
        except Exception:
            pass  # fall back to deterministic rationale on any LLM error

    node = make_node(
        agent="EligibilityTriage",
        summary=f"market_route={chosen.route.value}: {rationale}",
        payload={"branches": [b.model_dump() for b in branches]},
    )
    return {
        "market_route": chosen.route,
        "decision_trace": append_node(list(state.get("decision_trace", [])), node),
    }
