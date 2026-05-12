"""CitizensBenchmarkTool - FL Citizens 2026 actuarial benchmark per $1000 of Coverage A.

Reads ``data/tables/citizens_2026_rate_filing.csv``. Used by the Validator's
cohort/actuarial sanity check on FL quotes.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ._data import load_csv


class CitizensInput(BaseModel):
    state: str = Field(..., min_length=2, max_length=2)
    county_fips: str = Field(..., min_length=4, max_length=5)
    coastal_distance_band: Literal["coastal_0_5mi", "5_to_15mi", "inland_15plus"] = "coastal_0_5mi"


class CitizensOutput(BaseModel):
    state: str
    county_fips: str
    county_name: str
    coastal_distance_band: str
    hurricane_tier: str
    base_actuarial_per_1000_cov_a: float
    evidence_id: str
    annual_actuarial_at_cov_a_usd: float | None = None


@tool("citizens_benchmark", args_schema=CitizensInput)
def citizens_benchmark(
    state: str,
    county_fips: str,
    coastal_distance_band: str = "coastal_0_5mi",
) -> dict:
    """Return Citizens' 2026 multi-peril HO-3 benchmark in dollars per $1000 of Coverage A for an FL county.

    Returns ``base_actuarial_per_1000_cov_a`` only; the Validator multiplies by Coverage A / 1000 to get the annual figure.
    Returns NULL/None if the state is not FL (Citizens is FL-only).
    """
    rows = load_csv("citizens_2026_rate_filing.csv")
    state = state.upper()
    cf = county_fips.zfill(5)
    if state != "FL":
        raise ValueError("citizens_benchmark is FL-only; for non-FL flows use cohort_benchmark instead.")
    matching = [r for r in rows if r["state"] == state and r["county_fips"].zfill(5) == cf]
    if not matching:
        raise ValueError(f"citizens_benchmark: no row for state={state} county={cf}")
    chosen = next((r for r in matching if r["coastal_distance_band"] == coastal_distance_band), matching[0])
    return CitizensOutput(
        state=state,
        county_fips=chosen["county_fips"],
        county_name=chosen["county_name"],
        coastal_distance_band=chosen["coastal_distance_band"],
        hurricane_tier=chosen["hurricane_tier"],
        base_actuarial_per_1000_cov_a=float(chosen["base_actuarial_per_1000_cov_a"]),
        evidence_id=chosen["evidence_id"],
    ).model_dump()
