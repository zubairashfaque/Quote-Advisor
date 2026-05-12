"""ReplacementCostTool - estimate dwelling rebuild cost.

Reads ``data/tables/replacement_cost_factors.csv``. Synthetic-but-calibrated
locality factors (RSMeans / Marshall & Swift are proprietary); see
data/REAL_DATA_PROVENANCE.md.
"""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ._data import load_csv


class ReplacementCostInput(BaseModel):
    state: str = Field(..., min_length=2, max_length=2)
    zip_code: str | None = Field(default=None, description="5-digit US zip; first 3 used for locality factor lookup")
    sqft: float = Field(default=2400.0, ge=200.0, description="Living square footage; defaults to median single-family value")
    home_value_usd: float = Field(..., ge=0.0)


class ReplacementCostOutput(BaseModel):
    state: str
    zip3: str
    locality_factor: float
    base_cost_per_sqft: float
    rebuild_cost_usd: float
    home_value_usd: float
    rebuild_to_market_ratio: float
    evidence_id: str
    notes: str


@tool("replacement_cost", args_schema=ReplacementCostInput)
def replacement_cost(
    state: str,
    home_value_usd: float,
    zip_code: str | None = None,
    sqft: float = 2400.0,
) -> dict:
    """Estimate dwelling rebuild cost via locality_factor x base_cost_per_sqft x sqft.

    Used by the Coverage Agent (Step 3 - right-size limits) to feed Cov A floors and lender requirements.
    Falls back to the DEFAULT row when zip3 / state are unknown.
    """
    rows = load_csv("replacement_cost_factors.csv")
    state = state.upper()
    zip3 = (zip_code or "")[:3]
    chosen = None
    if zip3:
        for r in rows:
            if r["state"] == state and r["zip3_prefix"] == zip3:
                chosen = r
                break
    if chosen is None:
        for r in rows:
            if r["state"] == state and r["zip3_prefix"] != "DEFAULT":
                chosen = r
                break
    if chosen is None:
        chosen = next(r for r in rows if r["zip3_prefix"] == "DEFAULT")

    locality = float(chosen["locality_factor"])
    per_sqft = float(chosen["base_cost_per_sqft"])
    rebuild = locality * per_sqft * sqft
    ratio = (rebuild / home_value_usd) if home_value_usd > 0 else 0.0
    return ReplacementCostOutput(
        state=state,
        zip3=chosen["zip3_prefix"],
        locality_factor=locality,
        base_cost_per_sqft=per_sqft,
        rebuild_cost_usd=round(rebuild, 0),
        home_value_usd=home_value_usd,
        rebuild_to_market_ratio=round(ratio, 3),
        evidence_id=chosen["evidence_id"],
        notes=chosen["notes"],
    ).model_dump()
