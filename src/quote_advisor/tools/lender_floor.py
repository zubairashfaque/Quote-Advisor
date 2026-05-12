"""LenderFloorTool - Coverage A floor + form requirement from lender rules.

Reads ``data/tables/lender_minimums.json``. Implements Fannie Mae B7-3-02 +
Freddie Mac §4703.2 + FDPA 1973 (NFIP mandatory purchase).
"""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ._data import load_json


class LenderFloorInput(BaseModel):
    has_mortgage: bool = True
    in_sfha: bool = False
    home_value_usd: float = Field(..., ge=0.0)
    rebuild_cost_usd: float | None = Field(default=None, ge=0.0)
    unpaid_principal_balance_usd: float | None = Field(default=None, ge=0.0)


class LenderFloorOutput(BaseModel):
    has_mortgage: bool
    in_sfha: bool
    coverage_a_floor_usd: float
    settlement_basis: str
    form_required: str
    nfip_required: bool
    citations: list[str]
    evidence_ids: list[str]


@tool("lender_floor", args_schema=LenderFloorInput)
def lender_floor(
    has_mortgage: bool = True,
    in_sfha: bool = False,
    home_value_usd: float = 0.0,
    rebuild_cost_usd: float | None = None,
    unpaid_principal_balance_usd: float | None = None,
) -> dict:
    """Return the lender-imposed Coverage A floor and the form requirement.

    Per Fannie B7-3-02 / Freddie 4703.2: Coverage A >= min(replacement_cost, unpaid_principal_balance).
    NFIP mandatory if in_sfha AND has_mortgage (FDPA 1973).
    """
    rules = load_json("lender_minimums.json")
    citations: list[str] = []
    evidence_ids: list[str] = []
    coverage_a_floor: float = 0.0
    settlement = "actual_cash_value"
    form = "HO-3"
    nfip = False

    if has_mortgage:
        gse_rule = next(r for r in rules["rules"] if r["rule_id"] == "GSE-FANNIE-B7-3-02")
        rcv = rebuild_cost_usd if rebuild_cost_usd is not None else home_value_usd
        upb = unpaid_principal_balance_usd if unpaid_principal_balance_usd is not None else home_value_usd
        coverage_a_floor = max(coverage_a_floor, min(rcv, upb))
        settlement = gse_rule["settlement_basis"]
        form = gse_rule["form_required"]
        citations.append(gse_rule["citation"])
        evidence_ids.append(gse_rule["evidence_id"])

    if in_sfha and has_mortgage:
        nfip_rule = next(r for r in rules["rules"] if r["rule_id"] == "NFIP-MANDATORY-PURCHASE")
        nfip = True
        citations.append(nfip_rule["citation"])
        evidence_ids.append(nfip_rule["evidence_id"])

    if coverage_a_floor == 0.0:
        coverage_a_floor = home_value_usd

    return LenderFloorOutput(
        has_mortgage=has_mortgage,
        in_sfha=in_sfha,
        coverage_a_floor_usd=round(coverage_a_floor, 2),
        settlement_basis=settlement,
        form_required=form,
        nfip_required=nfip,
        citations=citations,
        evidence_ids=evidence_ids,
    ).model_dump()
