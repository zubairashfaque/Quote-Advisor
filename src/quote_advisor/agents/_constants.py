"""Module-level constants shared across agent modules.

Lives outside ``agents/__init__.py`` so individual agent modules can import
constants without triggering a circular re-import of the parent package.
"""

from __future__ import annotations

# Default lat/lon centroids per state - used when the customer profile lacks
# explicit coordinates. These mirror the seeded entries in the cached
# ``data/api_samples/usgs_pga_cache.json``.
DEFAULT_LATLON: dict[str, tuple[float, float]] = {
    "CA": (34.0522, -118.2437),  # Los Angeles
    "FL": (25.7617,  -80.1918),  # Miami
}
