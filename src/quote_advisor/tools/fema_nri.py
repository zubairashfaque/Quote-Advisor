"""FEMANrIRiskTool - county-level hazard tier lookup.

Reads ``data/tables/fema_nri_counties.csv`` (curated subset of the FEMA NRI
release; refresh via ``data/scripts/fetch_real_data.py --only fema_nri``).
"""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ._data import load_csv


class FEMANrIInput(BaseModel):
    county_fips: str = Field(..., min_length=4, max_length=5, description="5-digit county FIPS code (zero-pad single-digit states)")


class FEMANrIOutput(BaseModel):
    county_fips: str
    state: str
    county_name: str
    overall_score: float
    overall_tier: str
    eal_total_usd: float
    eal_wildfire_usd: float
    eal_hurricane_usd: float
    eal_earthquake_usd: float
    eal_flood_usd: float
    evidence_id: str


@tool("fema_nri_risk", args_schema=FEMANrIInput)
def fema_nri_risk(county_fips: str) -> dict:
    """Look up the FEMA National Risk Index aggregate score and per-peril Expected Annual Loss for a US county.

    Returns overall_score (0-100), overall_tier (e.g., 'Very High'), and per-peril EAL in USD.
    Use this for the Risk Agent's macro-tier signal; combine with finer tools (FHSZ, NFHL, USGS, HURDAT) for granular per-peril severity.
    """
    rows = load_csv("fema_nri_counties.csv")
    fips = county_fips.zfill(5)
    for row in rows:
        if row["county_fips"].zfill(5) == fips:
            return FEMANrIOutput(
                county_fips=row["county_fips"],
                state=row["state"],
                county_name=row["county_name"],
                overall_score=float(row["overall_score"]),
                overall_tier=row["overall_tier"],
                eal_total_usd=float(row["eal_total_usd"]),
                eal_wildfire_usd=float(row["eal_wildfire_usd"]),
                eal_hurricane_usd=float(row["eal_hurricane_usd"]),
                eal_earthquake_usd=float(row["eal_earthquake_usd"]),
                eal_flood_usd=float(row["eal_flood_usd"]),
                evidence_id=row["evidence_id"],
            ).model_dump()
    raise ValueError(f"FEMA NRI: no curated row for county_fips={fips}. Refresh via data/scripts/fetch_real_data.py --only fema_nri.")
