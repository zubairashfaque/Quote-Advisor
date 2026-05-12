"""NOAAHurricaneTool - per-county hurricane exposure tier from NOAA HURDAT2.

Reads ``data/tables/hurricane_exposure_tiers.csv`` (derived from HURDAT2 raw).
Refresh raw via ``data/scripts/fetch_real_data.py --only hurdat`` then re-aggregate.
"""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ._data import load_csv


class NOAAHurricaneInput(BaseModel):
    state: str = Field(..., min_length=2, max_length=2, description="2-letter state code (e.g., FL)")
    county_fips: str | None = Field(default=None, description="5-digit county FIPS (optional). When omitted, returns the state's worst-tier county hint.")


class NOAAHurricaneOutput(BaseModel):
    state: str
    county_fips: str
    county_name: str
    hurricane_tier: str
    landfalls_within_75mi_since_1900: int
    strongest_landfall_category: int
    evidence_id: str


def _pick_default(rows: list[dict[str, str]], state: str) -> dict[str, str] | None:
    matching = [r for r in rows if r["state"] == state]
    if not matching:
        return None
    return max(matching, key=lambda r: int(r.get("landfalls_within_75mi_since_1900", "0")))


@tool("noaa_hurricane", args_schema=NOAAHurricaneInput)
def noaa_hurricane(state: str, county_fips: str | None = None) -> dict:
    """Return per-county hurricane exposure tier and HURDAT2 landfall counts.

    For non-Atlantic-coast states (CA, etc.) the tier is always 'Low' and landfalls=0.
    Use the optional ``county_fips`` for precision; otherwise the state's worst-tier county is used as the default.
    """
    rows = load_csv("hurricane_exposure_tiers.csv")
    state = state.upper()
    chosen: dict[str, str] | None = None
    if county_fips:
        cf = county_fips.zfill(5)
        for r in rows:
            if r["state"] == state and r["county_fips"].zfill(5) == cf:
                chosen = r
                break
    if chosen is None:
        chosen = _pick_default(rows, state)
    if chosen is None:
        for r in rows:
            if r["state"] == "DEFAULT":
                chosen = r
                break
    if chosen is None:
        raise RuntimeError("noaa_hurricane: no rows in hurricane_exposure_tiers.csv (and no DEFAULT row).")

    return NOAAHurricaneOutput(
        state=state,
        county_fips=chosen.get("county_fips", county_fips or ""),
        county_name=chosen.get("county_name", "Unknown"),
        hurricane_tier=chosen["hurricane_tier"],
        landfalls_within_75mi_since_1900=int(chosen.get("landfalls_within_75mi_since_1900", "0")),
        strongest_landfall_category=int(chosen.get("strongest_landfall_category", "0")),
        evidence_id=chosen["evidence_id"],
    ).model_dump()
