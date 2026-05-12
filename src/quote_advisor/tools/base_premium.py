"""BasePremiumTool - state/year base premium lookup.

Reads ``data/tables/pricing_benchmarks_2025_2026.csv``. Anchored on NAIC 2022
and Bankrate 2025 published averages; see data/REAL_DATA_PROVENANCE.md.
"""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ._data import load_csv


class BasePremiumInput(BaseModel):
    state: str = Field(..., min_length=2, max_length=2)
    year: int = Field(default=2026, ge=2020, le=2030)


class BasePremiumOutput(BaseModel):
    state: str
    year: int
    base_premium_usd: float
    scope: str
    source: str
    evidence_id: str


@tool("base_premium", args_schema=BasePremiumInput)
def base_premium(state: str, year: int = 2026) -> dict:
    """Return the state-level annual home insurance base premium for a given year.

    Returned premium is the HO-3 statewide median for a 250K dwelling reference; the Pricing Agent's home_value_scaling multiplier adjusts for actual dwelling.
    Falls back to closest-year row when an exact year is missing.
    """
    rows = load_csv("pricing_benchmarks_2025_2026.csv")
    state = state.upper()
    matching = [r for r in rows if r["state"].upper() == state]
    if not matching:
        raise ValueError(f"base_premium: no benchmarks for state={state}")
    exact = [r for r in matching if int(r["year"]) == year]
    if exact:
        row = exact[0]
    else:
        row = min(matching, key=lambda r: abs(int(r["year"]) - year))
    return BasePremiumOutput(
        state=row["state"],
        year=int(row["year"]),
        base_premium_usd=float(row["base_premium_usd"]),
        scope=row["scope"],
        source=row["source"],
        evidence_id=row["evidence_id"],
    ).model_dump()
