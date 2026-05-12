"""RiskFactorTool - aggregator that scores a profile across the Risk Agent's surface.

Convenience tool: composes hazard tools into a single per-profile RiskFactor
breakdown. The Risk Agent's ReAct loop typically calls the underlying hazard
tools individually; this tool is offered as a one-shot aggregator for
deterministic flows or for the cohort/Counterfactual paths where the LLM
should not re-discover hazards.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from . import (
    ca_fire_zone,
    flood_zone,
    noaa_hurricane,
    usgs_seismic,
)


class RiskFactorAggInput(BaseModel):
    state: str = Field(..., min_length=2, max_length=2)
    lat: float
    lon: float
    has_pool: bool = False
    claims_history: int = 0
    county_fips: str | None = None


class RiskFactorScored(BaseModel):
    factor: str
    severity: Literal["low", "medium", "high"]
    rationale: str
    evidence_ids: list[str]


class RiskFactorAggOutput(BaseModel):
    factors: list[RiskFactorScored]


def _tier_to_sev(tier: str) -> Literal["low", "medium", "high"]:
    t = (tier or "").strip().lower()
    if t in {"very high", "high"}:
        return "high"
    if t in {"moderate"}:
        return "medium"
    return "low"


@tool("risk_factor_breakdown", args_schema=RiskFactorAggInput)
def risk_factor_breakdown(
    state: str,
    lat: float,
    lon: float,
    has_pool: bool = False,
    claims_history: int = 0,
    county_fips: str | None = None,
) -> dict:
    """Aggregate hazard tools into a single per-profile risk-factor list.

    Returns a list of {factor, severity, rationale, evidence_ids} ready to feed the QuoteOutput.risk_factors field.
    """
    state = state.upper()
    out: list[RiskFactorScored] = []

    # Wildfire (CA only - FL profiles get an explicit 'low' here).
    if state == "CA":
        fhsz = ca_fire_zone.ca_fire_zone.invoke({"lat": lat, "lon": lon})
        out.append(
            RiskFactorScored(
                factor="Wildfire",
                severity=_tier_to_sev(fhsz["fhsz_tier"]),
                rationale=f"CAL FIRE FHSZ tier {fhsz['fhsz_tier']} at lat/lon (nearest reference: {fhsz['nearest_reference_city']}).",
                evidence_ids=[fhsz["evidence_id"]],
            )
        )

    # Seismic (relevant only when PGA >= 0.20g).
    seismic = usgs_seismic.usgs_seismic.invoke({"lat": lat, "lon": lon})
    if (seismic.get("pga_g") or 0) >= 0.20:
        out.append(
            RiskFactorScored(
                factor="Seismic",
                severity=_tier_to_sev(seismic["tier"]),
                rationale=f"USGS PGA {seismic['pga_g']}g (tier {seismic['tier']}).",
                evidence_ids=[seismic["evidence_id"]],
            )
        )

    # Hurricane (FL primary).
    if state == "FL":
        hurr = noaa_hurricane.noaa_hurricane.invoke({"state": state, "county_fips": county_fips})
        out.append(
            RiskFactorScored(
                factor="Hurricane",
                severity=_tier_to_sev(hurr["hurricane_tier"]),
                rationale=f"NOAA HURDAT2: {hurr['landfalls_within_75mi_since_1900']} landfalls within 75mi since 1900 in {hurr['county_name']}.",
                evidence_ids=[hurr["evidence_id"]],
            )
        )

    # Flood (universal).
    flood = flood_zone.flood_zone.invoke({"lat": lat, "lon": lon})
    if flood.get("in_sfha"):
        out.append(
            RiskFactorScored(
                factor="Flood",
                severity="high",
                rationale=f"FEMA NFHL zone {flood['flood_zone']} (Special Flood Hazard Area, BFE {flood.get('bfe_ft')}ft).",
                evidence_ids=[flood["evidence_id"]],
            )
        )

    # Pool liability.
    if has_pool:
        out.append(
            RiskFactorScored(
                factor="Pool liability",
                severity="medium",
                rationale="On-premises pool flagged as attractive nuisance; liability factor uplifted.",
                evidence_ids=["MULT-POOL-HIGH"],
            )
        )

    # Claims history.
    if claims_history >= 1:
        out.append(
            RiskFactorScored(
                factor="Claims history",
                severity="medium" if claims_history == 1 else "high",
                rationale=f"{claims_history} prior claim(s) on record; pricing claims-multiplier applies.",
                evidence_ids=[f"MULT-CLAIMS-{claims_history if claims_history < 3 else '3PLUS'}"],
            )
        )

    return RiskFactorAggOutput(factors=out).model_dump()
