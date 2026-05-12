# `data/geo/` — Geospatial reference data

## `calfire_fhsz.geojson` (not tracked)

CAL FIRE Fire Hazard Severity Zones (SRA + LRA combined) polygon feature collection — used by `src/quote_advisor/tools/ca_fire_zone.py` to classify a California property's lat/lon into Moderate / High / Very High fire-hazard zones.

The file is **218 MB**, which exceeds GitHub's 100 MB per-file hard limit, so it is excluded from version control via `.gitignore`. To enable the `ca_fire_zone` tool after cloning, re-download it from the official CAL FIRE FRAP source:

- **Portal:** https://frap.fire.ca.gov/mapping/gis-data/
- **Layer:** Fire Hazard Severity Zones (FHSZ) — combined SRA & LRA
- **Format:** GeoJSON (Esri ArcGIS REST → query as GeoJSON, or use the `data/scripts/fetch_real_data.py` helper which fetches the same layer via the ArcGIS REST API)

Expected target path after download: `data/geo/calfire_fhsz.geojson`.

The provenance ledger at `data/REAL_DATA_PROVENANCE.md` records the exact ArcGIS REST endpoint and the date of the last refresh.
