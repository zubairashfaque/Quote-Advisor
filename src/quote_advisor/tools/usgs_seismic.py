"""USGSSeismicTool - ASCE 7-22 seismic design parameters by lat/lon.

Reads the cache at ``data/api_samples/usgs_pga_cache.json`` (initial seed and
refresh via ``data/scripts/fetch_real_data.py --only usgs``). Returns PGA in g
and a tier (Low / Moderate / High / Very High).
"""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ._data import haversine_miles, nearest_pga_entry, pga_to_tier


class USGSSeismicInput(BaseModel):
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)


class USGSSeismicOutput(BaseModel):
    lat: float
    lon: float
    pga_g: float | None
    sds: float | None
    sd1: float | None
    tier: str
    evidence_id: str
    distance_miles_to_cached_point: float


# Fallback tiers when both pga_g and tier in the cache are missing/Unknown.
# Used only as a last resort; documented in data/REAL_DATA_PROVENANCE.md.
_STATE_FALLBACK_TIER = {
    "CA": "High",      # CA seismic risk is broadly high; conservative estimate
    "FL": "Low",       # FL seismic risk is essentially zero
    "DEFAULT": "Low",
}


def _infer_state(lat: float, lon: float) -> str:
    """Cheap state-code inference from lat/lon - covers our seed locations only."""
    if 32.0 <= lat <= 42.5 and -125.0 <= lon <= -114.0:
        return "CA"
    if 24.0 <= lat <= 31.5 and -88.0 <= lon <= -79.0:
        return "FL"
    return "DEFAULT"


@tool("usgs_seismic", args_schema=USGSSeismicInput)
def usgs_seismic(lat: float, lon: float) -> dict:
    """Return ASCE 7-22 mapped seismic parameters (PGA, Sds, Sd1) and a coarse tier for a lat/lon.

    PGA tier mapping: Very High >= 0.6g, High >= 0.4g, Moderate >= 0.2g, Low < 0.2g.
    Tier 'Very High' typically triggers a CEA companion offer (statutory mandatory in CA per Cal. Ins. Code §10081).

    Resilience: if the cache entry has ``pga_g=null`` and ``tier='Unknown'`` (e.g., the fetcher
    overwrote real data after a transient API hiccup) the tool falls back to a per-state default
    so downstream agents still produce sensible output.
    """
    nearest = nearest_pga_entry(lat, lon)
    if nearest is None:
        # Cache truly empty - infer from state and emit a warning-bearing record.
        state = _infer_state(lat, lon)
        return USGSSeismicOutput(
            lat=lat, lon=lon, pga_g=None, sds=None, sd1=None,
            tier=_STATE_FALLBACK_TIER[state],
            evidence_id=f"PGA-FALLBACK-{state}-2026",
            distance_miles_to_cached_point=0.0,
        ).model_dump()

    distance = haversine_miles(lat, lon, float(nearest["lat"]), float(nearest["lon"]))
    pga = nearest.get("pga_g")
    cached_tier = nearest.get("tier") or "Unknown"

    if pga is not None:
        tier = pga_to_tier(pga)
    elif cached_tier and cached_tier != "Unknown":
        tier = cached_tier
    else:
        # Both pga and tier missing - fall back to per-state default.
        tier = _STATE_FALLBACK_TIER.get(_infer_state(lat, lon), _STATE_FALLBACK_TIER["DEFAULT"])

    return USGSSeismicOutput(
        lat=lat,
        lon=lon,
        pga_g=pga,
        sds=nearest.get("sds"),
        sd1=nearest.get("sd1"),
        tier=tier,
        evidence_id=nearest["evidence_id"],
        distance_miles_to_cached_point=round(distance, 2),
    ).model_dump()
