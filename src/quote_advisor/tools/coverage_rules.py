"""CoverageRulesTool - statutory minimums and required offers per state.

Composes the StatutoryRulesEngine output with state-specific defaults the
Coverage Agent needs (e.g., FL CGCC mandatory, CA Loss-of-Use 24mo floor).
"""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .. import statutory_rules_engine


class CoverageRulesInput(BaseModel):
    state: str = Field(..., min_length=2, max_length=2)
    home_value_usd: float = Field(..., ge=0.0)
    has_mortgage: bool = False
    in_sfha: bool = False


class CoverageRulesOutput(BaseModel):
    state: str
    required_coverages: list[str]
    required_offers: list[dict]
    required_forms: list[str]
    floors: dict
    market_route_hints: list[str]
    triggered_rule_ids: list[str]


@tool("coverage_rules", args_schema=CoverageRulesInput)
def coverage_rules(
    state: str,
    home_value_usd: float,
    has_mortgage: bool = False,
    in_sfha: bool = False,
) -> dict:
    """Return statutory minimums, required offers, required forms, and floors for a state.

    Wraps the pre-LLM StatutoryRulesEngine; the Coverage Agent's Plan-and-Execute planner consumes the result as Step 1 (floors) and Step 4 (endorsements).
    """
    raw = {
        "age": 50,  # placeholder - rules don't depend on age beyond CA-AGE-NON-PRIMARY
        "location": state,
        "home_value": int(home_value_usd),
        "has_pool": False,
        "claims_history": 0,
        "credit_score": None,
    }
    out = statutory_rules_engine.apply(raw, context={"has_mortgage": has_mortgage, "in_sfha": in_sfha})
    return CoverageRulesOutput(
        state=state.upper(),
        required_coverages=out.required_coverages,
        required_offers=out.required_offers,
        required_forms=out.required_forms,
        floors=out.floors,
        market_route_hints=out.market_route_hints,
        triggered_rule_ids=[r.rule_id for r in out.triggered_rules],
    ).model_dump()
