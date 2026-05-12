"""CEAEarthquakeRecommenderTool - recommend a CA Earthquake Authority deductible.

Reads ``data/tables/cea_deductible_rules.json``. Statutorily mandatory offer in
CA per Cal. Ins. Code §10081.
"""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ._data import load_json


class CEAInput(BaseModel):
    coverage_a_usd: float = Field(..., ge=0.0)
    year_built: int | None = Field(default=None, ge=1800, le=2100)
    foundation: str | None = Field(default=None, description="slab | perimeter | crawlspace | post-and-beam | unknown")


class CEAOutput(BaseModel):
    deductible_options_pct: list[int]
    recommended_deductible_pct: int
    recommended_deductible_usd: float
    rationale: str
    premium_factor_at_recommendation: float
    evidence_id: str


def _evaluate_condition(cond: str, year_built: int | None, foundation: str | None) -> bool:
    if cond == "default":
        return False
    foundation = (foundation or "unknown").strip().lower()
    if "year_built >= 1980" in cond and (year_built is None or year_built < 1980):
        return False
    if "year_built >= 1960" in cond:
        if year_built is None or year_built < 1960:
            return False
        if "year_built < 1980" in cond and year_built >= 1980:
            return False
    if "year_built < 1960" in cond:
        if year_built is not None and year_built >= 1960:
            return "OR" in cond and foundation == "unknown"
        return True
    if "foundation in" in cond:
        if foundation not in {"slab", "perimeter"}:
            return False
    if "foundation == 'unknown'" in cond:
        if foundation != "unknown":
            return False
    return True


@tool("cea_earthquake_recommender", args_schema=CEAInput)
def cea_earthquake_recommender(
    coverage_a_usd: float,
    year_built: int | None = None,
    foundation: str | None = None,
) -> dict:
    """Recommend a CEA earthquake-deductible percentage based on construction era and foundation type.

    Returns the percent options (5/10/15/20/25), the recommendation, the dollar deductible at that percent, and the premium factor that would apply.
    """
    rules = load_json("cea_deductible_rules.json")
    options: list[int] = list(rules["deductible_options_pct"])
    factors: dict[str, float] = {str(k): float(v) for k, v in rules["premium_factor_per_deductible_pct"].items()}

    chosen_pct: int | None = None
    rationale: str = ""
    for cond_block in rules["default_recommendation_logic"]:
        if cond_block["condition"] == "default":
            continue
        if _evaluate_condition(cond_block["condition"], year_built, foundation):
            chosen_pct = int(cond_block["recommended_deductible_pct"])
            rationale = cond_block["rationale"]
            break
    if chosen_pct is None:
        default_block = next(b for b in rules["default_recommendation_logic"] if b["condition"] == "default")
        chosen_pct = int(default_block["recommended_deductible_pct"])
        rationale = default_block["rationale"]

    return CEAOutput(
        deductible_options_pct=options,
        recommended_deductible_pct=chosen_pct,
        recommended_deductible_usd=round(coverage_a_usd * chosen_pct / 100.0, 2),
        rationale=rationale,
        premium_factor_at_recommendation=factors[str(chosen_pct)],
        evidence_id=rules["evidence_id"],
    ).model_dump()
