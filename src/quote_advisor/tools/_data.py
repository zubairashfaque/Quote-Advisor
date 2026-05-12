"""Shared data-table loaders for tool modules.

Caches table reads in-process so repeated tool calls don't re-parse CSVs.
Tables live in ``data/tables/`` (curated) and ``data/api_samples/`` (raw).
"""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..configuration import REPO_ROOT

TABLES_DIR = REPO_ROOT / "data" / "tables"
API_SAMPLES_DIR = REPO_ROOT / "data" / "api_samples"


@lru_cache(maxsize=None)
def load_csv(name: str, *, dir_: str = "tables") -> list[dict[str, str]]:
    """Load a CSV under ``data/<dir>/<name>`` as a list of row-dicts."""
    base = TABLES_DIR if dir_ == "tables" else API_SAMPLES_DIR
    path = base / name
    with path.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return rows


@lru_cache(maxsize=None)
def load_json(name: str, *, dir_: str = "tables") -> dict[str, Any]:
    """Load a JSON file under ``data/<dir>/<name>``."""
    base = TABLES_DIR if dir_ == "tables" else API_SAMPLES_DIR
    path = base / name
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in miles."""
    from math import asin, cos, radians, sin, sqrt

    r_miles = 3958.756
    p1, p2 = radians(lat1), radians(lat2)
    dp = radians(lat2 - lat1)
    dl = radians(lon2 - lon1)
    a = sin(dp / 2) ** 2 + cos(p1) * cos(p2) * sin(dl / 2) ** 2
    return 2 * r_miles * asin(sqrt(a))


def nearest_pga_entry(lat: float, lon: float) -> dict[str, Any] | None:
    """Find the nearest USGS PGA cache entry for the given lat/lon.

    Used by USGSSeismicTool to honour the "real fetched first, fall back to
    nearest cached point" pattern when the live API is offline.
    """
    cache = load_json("usgs_pga_cache.json", dir_="api_samples")
    entries = cache.get("entries", [])
    if not entries:
        return None
    best = min(
        entries,
        key=lambda e: haversine_miles(lat, lon, float(e["lat"]), float(e["lon"])),
    )
    return best


def pga_to_tier(pga: float | None) -> str:
    """USGS PGA -> tier (V4/V5 tier scheme)."""
    if pga is None:
        return "Unknown"
    if pga >= 0.60:
        return "Very High"
    if pga >= 0.40:
        return "High"
    if pga >= 0.20:
        return "Moderate"
    return "Low"
