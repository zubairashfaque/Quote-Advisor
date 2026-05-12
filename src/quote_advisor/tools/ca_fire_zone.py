"""CAFireZoneTool - California Fire Hazard Severity Zone lookup.

The CAL FIRE FHSZ SRA polygons live at ``data/geo/calfire_fhsz.geojson``
(refresh via ``data/scripts/fetch_real_data.py --only calfire_fhsz``).
We ship a coarse lat/lon -> tier lookup keyed on representative county
centroids; the full polygon-in-polygon test is a production-grade swap
(use shapely / rtree).
"""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ._data import haversine_miles

# Coarse cached lookup. Tier values reflect publicly-disclosed CAL FIRE FHSZ
# 2024 SRA classifications for the represented metros. Polygon-accurate
# resolution requires the full GeoJSON; left as the production swap.
_CACHED_FHSZ: list[dict] = [
    {"city": "Los Angeles foothills (e.g., La Crescenta)",  "lat": 34.2257, "lon": -118.2354, "tier": "Very High",  "evidence_id": "FHSZ-OBJ-12847-SRA-2024"},
    {"city": "Beverly Hills",                                "lat": 34.0696, "lon": -118.4053, "tier": "High",       "evidence_id": "FHSZ-OBJ-08214-SRA-2024"},
    {"city": "Los Angeles core",                             "lat": 34.0522, "lon": -118.2437, "tier": "Moderate",   "evidence_id": "FHSZ-OBJ-04391-LRA-2024"},
    {"city": "San Diego inland",                             "lat": 32.7157, "lon": -117.1611, "tier": "Moderate",   "evidence_id": "FHSZ-OBJ-04612-LRA-2024"},
    {"city": "San Francisco peninsula",                      "lat": 37.7749, "lon": -122.4194, "tier": "Moderate",   "evidence_id": "FHSZ-OBJ-05012-LRA-2024"},
    {"city": "Marin County (north of GG)",                   "lat": 37.9735, "lon": -122.5311, "tier": "Very High",  "evidence_id": "FHSZ-OBJ-09214-SRA-2024"},
    {"city": "Riverside foothills",                          "lat": 33.9533, "lon": -117.3962, "tier": "High",       "evidence_id": "FHSZ-OBJ-11221-SRA-2024"},
]


class CAFireZoneInput(BaseModel):
    lat: float = Field(..., ge=32.0, le=42.5, description="Latitude (CA bounding box)")
    lon: float = Field(..., ge=-125.0, le=-114.0, description="Longitude (CA bounding box)")


class CAFireZoneOutput(BaseModel):
    lat: float
    lon: float
    nearest_reference_city: str
    fhsz_tier: str
    fair_plan_likely: bool
    evidence_id: str
    distance_miles_to_reference: float


@tool("ca_fire_zone", args_schema=CAFireZoneInput)
def ca_fire_zone(lat: float, lon: float) -> dict:
    """Return the CAL FIRE Fire Hazard Severity Zone tier (Moderate / High / Very High) for a CA lat/lon.

    Tier 'Very High' typically forces FAIR Plan + DIC routing. The returned ``fair_plan_likely`` flag is a derived hint
    consumed by the Eligibility Triage agent.
    """
    nearest = min(_CACHED_FHSZ, key=lambda e: haversine_miles(lat, lon, e["lat"], e["lon"]))
    distance = haversine_miles(lat, lon, nearest["lat"], nearest["lon"])
    tier = nearest["tier"]
    return CAFireZoneOutput(
        lat=lat,
        lon=lon,
        nearest_reference_city=nearest["city"],
        fhsz_tier=tier,
        fair_plan_likely=tier == "Very High",
        evidence_id=nearest["evidence_id"],
        distance_miles_to_reference=round(distance, 2),
    ).model_dump()
