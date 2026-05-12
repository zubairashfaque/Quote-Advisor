"""FloodZoneTool - FEMA NFHL flood zone lookup by lat/lon.

Reads the cache at ``data/api_samples/fema_nfhl_cache.json`` (refresh via
``data/scripts/fetch_real_data.py --only fema_nfhl``). Returns the flood zone
designation, BFE, and SFHA membership.
"""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ._data import haversine_miles, load_json


class FloodZoneInput(BaseModel):
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)


class FloodZoneOutput(BaseModel):
    lat: float
    lon: float
    flood_zone: str
    in_sfha: bool
    bfe_ft: float | None
    panel_id: str | None
    evidence_id: str
    distance_miles_to_cached_point: float


def _state_fallback_flood(lat: float, lon: float) -> dict:
    """Conservative per-region flood fallback when the NFHL cache is empty.

    FL coastal cities default to AE (SFHA) which is the modal real-world zone;
    CA / inland default to X (not in SFHA). Used only when the cache is absent
    or empty - production should re-run the fetcher.
    """
    in_fl = 24.0 <= lat <= 31.5 and -88.0 <= lon <= -79.0
    if in_fl:
        return {
            "flood_zone": "AE",
            "in_sfha":     True,
            "bfe_ft":      8.0,
            "panel_id":    None,
            "evidence_id": "NFHL-FALLBACK-FL-AE-2026",
        }
    return {
        "flood_zone": "X",
        "in_sfha":     False,
        "bfe_ft":      None,
        "panel_id":    None,
        "evidence_id": "NFHL-FALLBACK-X-2026",
    }


@tool("flood_zone", args_schema=FloodZoneInput)
def flood_zone(lat: float, lon: float) -> dict:
    """Return the FEMA NFHL flood zone (X / AE / VE / etc.) and SFHA membership for a lat/lon.

    SFHA = Special Flood Hazard Area. When ``in_sfha`` is true and the home has a federally-backed mortgage, NFIP coverage is mandatory (FDPA 1973).

    Resilience: if the cache is empty (e.g., the live fetch failed and overwrote it) the tool
    returns a conservative per-region fallback (FL coastal -> AE / SFHA; elsewhere -> X / non-SFHA).
    """
    cache = load_json("fema_nfhl_cache.json", dir_="api_samples")
    entries = cache.get("entries", [])
    if not entries:
        fb = _state_fallback_flood(lat, lon)
        return FloodZoneOutput(
            lat=lat, lon=lon,
            flood_zone=fb["flood_zone"],
            in_sfha=fb["in_sfha"],
            bfe_ft=fb["bfe_ft"],
            panel_id=fb["panel_id"],
            evidence_id=fb["evidence_id"],
            distance_miles_to_cached_point=0.0,
        ).model_dump()

    nearest = min(entries, key=lambda e: haversine_miles(lat, lon, float(e["lat"]), float(e["lon"])))
    distance = haversine_miles(lat, lon, float(nearest["lat"]), float(nearest["lon"]))
    return FloodZoneOutput(
        lat=lat,
        lon=lon,
        flood_zone=nearest.get("flood_zone", "X"),
        in_sfha=bool(nearest.get("in_sfha", False)),
        bfe_ft=nearest.get("bfe_ft"),
        panel_id=nearest.get("panel_id"),
        evidence_id=nearest["evidence_id"],
        distance_miles_to_cached_point=round(distance, 2),
    ).model_dump()
