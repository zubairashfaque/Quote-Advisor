"""FLHurricaneDeductibleTool - the four statutory FL hurricane deductible options.

Reads ``data/tables/fl_hurricane_deductible_options.json``. Per Fla. Stat.
§627.701 the insurer must offer a flat $500 (when dwelling <= $250K) and 2/5/10%
of Coverage A. Coverage Agent must surface all four.
"""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ._data import load_json


class FLHurricaneInput(BaseModel):
    coverage_a_usd: float = Field(..., ge=0.0)


class FLHurricaneOption(BaseModel):
    option_id: str
    kind: str
    deductible_usd: float
    deductible_label: str
    premium_factor: float
    is_default: bool
    is_eligible: bool
    evidence_id: str


class FLHurricaneOutput(BaseModel):
    coverage_a_usd: float
    options: list[FLHurricaneOption]
    citation: str
    evidence_id: str


@tool("fl_hurricane_deductible", args_schema=FLHurricaneInput)
def fl_hurricane_deductible(coverage_a_usd: float) -> dict:
    """Return all four FL statutory hurricane deductible options computed at the given Coverage A.

    Each option includes the dollar deductible, premium-factor adjustment, and whether it is statutorily eligible (e.g., $500 flat is restricted to dwellings <= $250K).
    """
    rules = load_json("fl_hurricane_deductible_options.json")
    options: list[FLHurricaneOption] = []
    for opt in rules["options"]:
        eligible = True
        if opt["kind"] == "flat":
            ded = float(opt["amount_usd"])
            label = f"${int(ded):,}"
            if "applies_when_dwelling_le_usd" in opt:
                eligible = coverage_a_usd <= float(opt["applies_when_dwelling_le_usd"])
        else:
            pct = float(opt["percent"])
            ded = round(coverage_a_usd * pct / 100.0, 2)
            label = f"{int(pct)}% of Coverage A (${int(ded):,})"
        options.append(
            FLHurricaneOption(
                option_id=opt["option_id"],
                kind=opt["kind"],
                deductible_usd=ded,
                deductible_label=label,
                premium_factor=float(opt["premium_factor"]),
                is_default=bool(opt.get("is_default", False)),
                is_eligible=eligible,
                evidence_id=opt["evidence_id"],
            )
        )
    return FLHurricaneOutput(
        coverage_a_usd=coverage_a_usd,
        options=options,
        citation=rules["citation"],
        evidence_id=rules["evidence_id"],
    ).model_dump()
