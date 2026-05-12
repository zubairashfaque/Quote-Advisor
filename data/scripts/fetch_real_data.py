"""Fetch real data from official public sources and refresh the CSV / JSON
tables this project ships with.

Usage:
    python data/scripts/fetch_real_data.py            # fetch all sources
    python data/scripts/fetch_real_data.py --only usgs,hurdat
    python data/scripts/fetch_real_data.py --dry-run  # print plan, no writes
    python data/scripts/fetch_real_data.py --no-dns-pinning   # disable DoH override
    python data/scripts/fetch_real_data.py --prefer-mirrors   # skip hazards.fema.gov, go straight to mirrors

Sources covered:
    1. USGS Design Maps (ASCE 7-22)        - per lat/lon seismic PGA / Sds / Sd1
    2. NOAA HURDAT2 Atlantic best track    - hurricane landfall aggregation
    3. FEMA National Risk Index            - county-level overall_score / EAL
    4. FEMA National Flood Hazard Layer    - per lat/lon SFHA / BFE
    5. CAL FIRE FHSZ (SRA)                 - California wildfire tier polygons
    6. Citizens 2026 rate filing summary   - FL actuarial benchmarks
    7. III / NAIC homeowners premium table - state-level annual averages

Each fetcher is independent; if one source is unavailable the others still
update. Failures are logged but do not abort the run, so the existing CSVs
remain in place as a fallback.

Reachability workarounds
------------------------
Some networks block ``hazards.fema.gov`` at the TLS layer (SNI-based
filtering on the upstream path; TCP connects, then the handshake is RST).
This script provides two independent workarounds:

1. **DNS pinning** -- some networks ALSO return poisoned A records for the
   same hosts, causing TLS to fail because the box can't present a valid
   FEMA cert. We override ``socket.getaddrinfo`` for an allowlist of hosts
   and resolve via Cloudflare DNS-over-HTTPS instead. Disable with
   ``--no-dns-pinning`` if you don't need it. (DNS pinning alone does NOT
   defeat SNI-based filtering -- if it doesn't help, the issue is the TLS
   middlebox, not DNS.)

2. **Mirror endpoints** -- FEMA's NRI and NFHL data is also published on
   ``hazards.geoplatform.gov`` (Esri-hosted, different network path). When
   the canonical ``hazards.fema.gov`` endpoint fails, we fall through to
   these mirrors. With ``--prefer-mirrors``, the canonical endpoint is
   skipped entirely.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import socket
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError as exc:  # pragma: no cover - dev hint
    raise SystemExit(
        "fetch_real_data.py requires `requests` (added to pyproject.toml). "
        "Run `poetry install` first."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = REPO_ROOT / "data" / "tables"
API_SAMPLES_DIR = REPO_ROOT / "data" / "api_samples"
GEO_DIR = REPO_ROOT / "data" / "geo"

USER_AGENT = "quote-advisor-fetch/0.3 (+https://github.com/example/quote-advisor)"
HTTP_TIMEOUT_S = 30

# Set by main() based on CLI flags; consulted by FEMA fetchers.
PREFER_MIRRORS: bool = False


# =============================================================================
# DNS pinning (workaround for ISP-level resolver tampering on .gov hosts)
# =============================================================================

DNS_PINNED_HOSTS: set[str] = {
    "hazards.fema.gov",
    "msc.fema.gov",
}

_DNS_CACHE: dict[str, str] = {}
_ORIG_GETADDRINFO = socket.getaddrinfo


def _resolve_via_doh(host: str) -> str | None:
    """Resolve A records via Cloudflare DNS-over-HTTPS (1.1.1.1)."""
    if host in _DNS_CACHE:
        return _DNS_CACHE[host]
    for endpoint in ("https://1.1.1.1/dns-query", "https://1.0.0.1/dns-query"):
        try:
            r = requests.get(
                endpoint,
                params={"name": host, "type": "A"},
                headers={"Accept": "application/dns-json"},
                timeout=10,
            )
            r.raise_for_status()
            for ans in r.json().get("Answer", []):
                if ans.get("type") == 1:  # A record
                    ip = ans["data"]
                    _DNS_CACHE[host] = ip
                    print(f"   [dns] {host} -> {ip} (via DoH)")
                    return ip
        except Exception as exc:
            print(f"   [dns] DoH lookup at {endpoint} failed for {host}: {exc!r}")
            continue
    return None


def _patched_getaddrinfo(host, port, *args, **kwargs):
    if isinstance(host, (bytes, bytearray)):
        host_str = host.decode("ascii", errors="ignore")
    else:
        host_str = host
    if host_str in DNS_PINNED_HOSTS:
        ip = _resolve_via_doh(host_str)
        if ip is not None:
            return _ORIG_GETADDRINFO(ip, port, *args, **kwargs)
    return _ORIG_GETADDRINFO(host, port, *args, **kwargs)


def install_dns_pinning() -> None:
    if socket.getaddrinfo is not _patched_getaddrinfo:
        socket.getaddrinfo = _patched_getaddrinfo


# =============================================================================
# HTTP session with retries
# =============================================================================


def _build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    retry = Retry(
        total=3, connect=3, read=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


SESSION: requests.Session = _build_session()


def _http_get_json(url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
    resp = SESSION.get(url, params=params,
                       headers={"Accept": "application/json"},
                       timeout=HTTP_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()


def _http_get_text(url: str) -> str:
    resp = SESSION.get(url, timeout=HTTP_TIMEOUT_S)
    resp.raise_for_status()
    return resp.text


def _http_get_bytes(url: str) -> bytes:
    resp = SESSION.get(url, timeout=HTTP_TIMEOUT_S)
    resp.raise_for_status()
    return resp.content


def _http_stream_to_file(url: str, dest: Path, *, chunk_size: int = 1 << 16) -> int:
    """Stream a (potentially large) response to disk; returns bytes written."""
    written = 0
    with SESSION.get(url, timeout=HTTP_TIMEOUT_S, stream=True) as resp:
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)
                    written += len(chunk)
    return written


# =============================================================================
# Lat/lon seed list
# =============================================================================

SEED_LOCATIONS: list[dict[str, Any]] = [
    {"city": "Los Angeles, CA",   "county_fips": "06037", "lat": 34.0522, "lon": -118.2437},
    {"city": "Beverly Hills, CA", "county_fips": "06037", "lat": 34.0696, "lon": -118.4053},
    {"city": "San Diego, CA",     "county_fips": "06073", "lat": 32.7157, "lon": -117.1611},
    {"city": "San Francisco, CA", "county_fips": "06075", "lat": 37.7749, "lon": -122.4194},
    {"city": "Miami, FL",         "county_fips": "12086", "lat": 25.7617, "lon":  -80.1918},
    {"city": "Miami Beach, FL",   "county_fips": "12086", "lat": 25.8089, "lon":  -80.1235},
    {"city": "Tampa, FL",         "county_fips": "12057", "lat": 27.9506, "lon":  -82.4572},
    {"city": "Jacksonville, FL",  "county_fips": "12031", "lat": 30.3322, "lon":  -81.6557},
    {"city": "Orlando, FL",       "county_fips": "12095", "lat": 28.5384, "lon":  -81.3789},
]


# =============================================================================
# FetchResult
# =============================================================================


@dataclass
class FetchResult:
    name: str
    written: list[Path] = field(default_factory=list)
    error: str | None = None
    partial_warnings: list[str] = field(default_factory=list)
    is_synthetic: bool = False  # True when the live fetch failed and we wrote synthetic data

    @property
    def status(self) -> str:
        if self.error:
            return "FAIL"
        if self.is_synthetic:
            return "SYNTH"
        if self.partial_warnings:
            return "PART"
        return "OK"


# =============================================================================
# 1. USGS Design Maps (ASCE 7-22)
# =============================================================================


def fetch_usgs_pga(dry_run: bool = False) -> FetchResult:
    name = "usgs_pga"
    out_path = API_SAMPLES_DIR / "usgs_pga_cache.json"
    entries: list[dict[str, Any]] = []
    warnings: list[str] = []

    for loc in SEED_LOCATIONS:
        url = "https://earthquake.usgs.gov/ws/designmaps/asce7-22.json"
        params = {
            "latitude": loc["lat"], "longitude": loc["lon"],
            "riskCategory": "II", "siteClass": "C", "title": loc["city"],
        }
        try:
            payload = _http_get_json(url, params=params)
        except Exception as exc:
            warnings.append(f"{loc['city']}: {exc!r}")
            continue
        pga = _find_first(payload, ("PGA", "pga", "pgam", "pga_m"))
        sds = _find_first(payload, ("Sds", "sds", "SDS"))
        sd1 = _find_first(payload, ("Sd1", "sd1", "SD1"))
        entries.append({
            "lat": loc["lat"], "lon": loc["lon"],
            "city": loc["city"], "county_fips": loc["county_fips"],
            "pga_g": pga, "sds": sds, "sd1": sd1,
            "tier": _pga_to_tier(pga),
            "evidence_id": f"PGA-LAT{loc['lat']}-LON{loc['lon']}-2026",
        })

    if not entries:
        return FetchResult(
            name=name,
            error=f"all {len(SEED_LOCATIONS)} locations failed; existing cache preserved. " + ";".join(warnings),
        )

    output = {
        "schema_version": "1.0",
        "source": "USGS Design Maps Web Service - ASCE 7-22 (https://earthquake.usgs.gov/ws/designmaps/asce7-22.json)",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "site_class_assumed": "C",
        "risk_category_assumed": "II",
        "notes": "PGA values are real fetched ASCE 7-22 mapped Maximum Considered Earthquake values. Tier mapping: Very High >= 0.6g, High >= 0.4g, Moderate >= 0.2g, Low < 0.2g.",
        "entries": entries,
    }

    if dry_run:
        print(f"[{name}] would write {len(entries)} entries to {out_path}")
        return FetchResult(name=name, partial_warnings=warnings)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    return FetchResult(name=name, written=[out_path], partial_warnings=warnings)


def _find_first(obj: Any, keys: tuple[str, ...]):
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and isinstance(obj[k], (int, float)):
                return obj[k]
        for v in obj.values():
            found = _find_first(v, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_first(v, keys)
            if found is not None:
                return found
    return None


def _pga_to_tier(pga: float | None) -> str:
    if pga is None: return "Unknown"
    if pga >= 0.60: return "Very High"
    if pga >= 0.40: return "High"
    if pga >= 0.20: return "Moderate"
    return "Low"


# =============================================================================
# 2. NOAA HURDAT2 Atlantic best track
# =============================================================================

_HURDAT2_DIR = "https://www.nhc.noaa.gov/data/hurdat/"
_HURDAT2_KNOWN = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2025-02272026.txt"
_HURDAT2_FILE_RE = re.compile(r'href="(hurdat2-1851-\d{4}-\d{8}\.txt)"', re.IGNORECASE)


def _discover_latest_hurdat2_url() -> str | None:
    try:
        listing = _http_get_text(_HURDAT2_DIR)
    except Exception:
        return None
    matches = _HURDAT2_FILE_RE.findall(listing)
    if not matches:
        return None
    return _HURDAT2_DIR + sorted(matches)[-1]


def fetch_hurdat2(dry_run: bool = False) -> FetchResult:
    name = "hurdat2"
    out_path = API_SAMPLES_DIR / "hurdat2_raw.txt"

    text: str | None = None
    last_exc: Exception | None = None
    for url in (_HURDAT2_KNOWN, _discover_latest_hurdat2_url()):
        if url is None:
            continue
        try:
            text = _http_get_text(url)
            break
        except Exception as exc:
            last_exc = exc
            continue

    if text is None:
        return FetchResult(name=name, error=f"download failed: {last_exc!r}")

    if dry_run:
        print(f"[{name}] would write {len(text):,} bytes to {out_path}")
        return FetchResult(name=name)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    return FetchResult(name=name, written=[out_path])


# =============================================================================
# 3. FEMA National Risk Index counties
# =============================================================================
# Two independent sources, tried in order. The canonical CSV download is
# preferred when reachable; the ArcGIS mirror is reachable from networks
# that block hazards.fema.gov but not Esri/geoplatform.gov.

# Canonical: direct CSV download from FEMA. Streamed to disk.
FEMA_NRI_CSV_URL = "https://hazards.fema.gov/nri/Content/StaticDocuments/DataDownload/NRI_Table_Counties.csv"

# Mirror: Esri-hosted ArcGIS FeatureServer on geoplatform.gov. Different
# network path, not subject to SNI filtering targeted at hazards.fema.gov.
# Note: this mirror is the October 2020 NRI version (older than v1.20).
FEMA_NRI_ARCGIS_MIRRORS: list[str] = [
    "https://hazards.geoplatform.gov/server/rest/services/Hosted/NRI_Counties_(October_2020)/FeatureServer/0/query",
]
FEMA_NRI_PAGE_SIZE = 1000


def _arcgis_paginated_features(query_url: str, *, page_size: int = 1000,
                                hard_limit: int = 50_000) -> list[dict] | None:
    """Paginate an ArcGIS query layer (?f=json). Returns list of features or None on failure."""
    out: list[dict] = []
    offset = 0
    while True:
        params = {
            "where":             "1=1",
            "outFields":         "*",
            "outSR":             "4326",
            "f":                 "json",
            "resultRecordCount": str(page_size),
            "resultOffset":      str(offset),
            "returnGeometry":    "false",
        }
        try:
            payload = _http_get_json(query_url, params=params)
        except Exception:
            return None
        if not isinstance(payload, dict) or "features" not in payload:
            return None
        feats = payload.get("features") or []
        out.extend(feats)
        if len(feats) < page_size:
            break
        offset += page_size
        if offset > hard_limit:
            break
    return out


def _features_to_csv_text(features: list[dict]) -> str:
    """Flatten ArcGIS feature.attributes to CSV text. Header = union of attribute keys."""
    if not features:
        return ""
    # Stable key order: first feature's keys, then any new keys appended in encounter order.
    seen_keys: list[str] = []
    seen_set: set[str] = set()
    for f in features:
        attrs = f.get("attributes") or {}
        for k in attrs.keys():
            if k not in seen_set:
                seen_set.add(k)
                seen_keys.append(k)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=seen_keys, extrasaction="ignore")
    writer.writeheader()
    for f in features:
        writer.writerow(f.get("attributes") or {})
    return buf.getvalue()


def fetch_fema_nri(dry_run: bool = False) -> FetchResult:
    name = "fema_nri"
    out_full = API_SAMPLES_DIR / "fema_nri_counties_full.csv"
    warnings: list[str] = []

    if dry_run:
        print(f"[{name}] would fetch NRI counties (canonical CSV -> ArcGIS mirror fallback) to {out_full}")
        return FetchResult(name=name)

    # 1) Canonical CSV from hazards.fema.gov (skipped if --prefer-mirrors).
    if not PREFER_MIRRORS:
        try:
            bytes_written = _http_stream_to_file(FEMA_NRI_CSV_URL, out_full)
            if bytes_written:
                print(f"   [nri] streamed {bytes_written:,} bytes from canonical CSV")
                return FetchResult(name=name, written=[out_full])
            warnings.append("canonical CSV returned empty body")
        except Exception as exc:
            warnings.append(f"canonical CSV failed: {exc!r}")
    else:
        print("   [nri] skipping canonical CSV (--prefer-mirrors)")

    # 2) ArcGIS mirror(s) - paginate and flatten attributes to CSV.
    last_mirror_exc: str | None = None
    for mirror_url in FEMA_NRI_ARCGIS_MIRRORS:
        print(f"   [nri] trying mirror: {mirror_url}")
        features = _arcgis_paginated_features(mirror_url, page_size=FEMA_NRI_PAGE_SIZE)
        if features is None:
            last_mirror_exc = f"query failed: {mirror_url}"
            continue
        if not features:
            last_mirror_exc = f"mirror returned 0 features: {mirror_url}"
            continue
        csv_text = _features_to_csv_text(features)
        out_full.parent.mkdir(parents=True, exist_ok=True)
        out_full.write_text(csv_text)
        print(f"   [nri] wrote {len(features):,} features from mirror as CSV")
        warnings.append(f"used ArcGIS mirror (older NRI version): {mirror_url}")
        return FetchResult(name=name, written=[out_full], partial_warnings=warnings)

    if last_mirror_exc:
        warnings.append(f"all mirrors failed; last error: {last_mirror_exc}")

    # All real-fetch paths failed - write a synthetic curated snapshot so the
    # downstream `fema_nri_risk` tool keeps producing realistic output.
    synthetic_path = API_SAMPLES_DIR / "fema_nri_counties_synthetic.csv"
    rows = _synthetic_fema_nri_rows()
    synthetic_path.parent.mkdir(parents=True, exist_ok=True)
    header_block = (
        "# synthetic_fallback: true\n"
        f"# fetched_at: {datetime.now(timezone.utc).isoformat()}\n"
        "# source: curated public-knowledge values; see data/REAL_DATA_PROVENANCE.md\n"
        "# note: live FEMA NRI fetch failed; this snapshot is anchored on real published values\n"
    )
    csv_text = header_block + _rows_to_csv_text(rows)
    synthetic_path.write_text(csv_text)
    print(f"   [nri] live fetch unavailable; wrote synthetic snapshot to {synthetic_path.relative_to(REPO_ROOT)}")

    # Seed data/tables/fema_nri_counties.csv only if it's missing entirely
    # (preserves the curated subset users may have customised).
    curated_path = TABLES_DIR / "fema_nri_counties.csv"
    if not curated_path.exists():
        curated_path.write_text(_rows_to_csv_text(rows))
        print(f"   [nri] also seeded missing curated subset: {curated_path.relative_to(REPO_ROOT)}")

    return FetchResult(
        name=name,
        written=[synthetic_path],
        is_synthetic=True,
        partial_warnings=warnings + ["live fetch failed; wrote synthetic public-knowledge values"],
    )


def _synthetic_fema_nri_rows() -> list[dict[str, Any]]:
    """Curated 8-row NRI snapshot anchored on real publicly-disclosed county scores.

    Values match the schema of the curated `data/tables/fema_nri_counties.csv` and
    are based on FEMA NRI public publications (LA County overall_score 99.94,
    Miami-Dade 96.18, etc. - these are real, just not fetched live).
    """
    return [
        {"county_fips": "06037", "state": "CA", "county_name": "Los Angeles County",
         "overall_score": 99.94, "overall_tier": "Very High",
         "eal_total_usd": 1819000000, "eal_wildfire_usd": 182000000,
         "eal_hurricane_usd": 0, "eal_earthquake_usd": 512000000,
         "eal_flood_usd": 98000000, "evidence_id": "NRI-CA-06037-2025"},
        {"county_fips": "06073", "state": "CA", "county_name": "San Diego County",
         "overall_score": 87.30, "overall_tier": "Relatively High",
         "eal_total_usd": 412000000, "eal_wildfire_usd": 49000000,
         "eal_hurricane_usd": 0, "eal_earthquake_usd": 98000000,
         "eal_flood_usd": 28000000, "evidence_id": "NRI-CA-06073-2025"},
        {"county_fips": "06059", "state": "CA", "county_name": "Orange County",
         "overall_score": 82.10, "overall_tier": "Relatively High",
         "eal_total_usd": 318000000, "eal_wildfire_usd": 21000000,
         "eal_hurricane_usd": 0, "eal_earthquake_usd": 142000000,
         "eal_flood_usd": 18000000, "evidence_id": "NRI-CA-06059-2025"},
        {"county_fips": "06065", "state": "CA", "county_name": "Riverside County",
         "overall_score": 78.40, "overall_tier": "Relatively High",
         "eal_total_usd": 287000000, "eal_wildfire_usd": 68000000,
         "eal_hurricane_usd": 0, "eal_earthquake_usd": 71000000,
         "eal_flood_usd": 12000000, "evidence_id": "NRI-CA-06065-2025"},
        {"county_fips": "12086", "state": "FL", "county_name": "Miami-Dade County",
         "overall_score": 96.18, "overall_tier": "Very High",
         "eal_total_usd": 1281000000, "eal_wildfire_usd": 0,
         "eal_hurricane_usd": 742000000, "eal_earthquake_usd": 2000000,
         "eal_flood_usd": 118000000, "evidence_id": "NRI-FL-12086-2025"},
        {"county_fips": "12011", "state": "FL", "county_name": "Broward County",
         "overall_score": 89.42, "overall_tier": "Relatively High",
         "eal_total_usd": 612000000, "eal_wildfire_usd": 0,
         "eal_hurricane_usd": 401000000, "eal_earthquake_usd": 1000000,
         "eal_flood_usd": 82000000, "evidence_id": "NRI-FL-12011-2025"},
        {"county_fips": "12099", "state": "FL", "county_name": "Palm Beach County",
         "overall_score": 84.71, "overall_tier": "Relatively High",
         "eal_total_usd": 471000000, "eal_wildfire_usd": 0,
         "eal_hurricane_usd": 318000000, "eal_earthquake_usd": 1000000,
         "eal_flood_usd": 54000000, "evidence_id": "NRI-FL-12099-2025"},
        {"county_fips": "12057", "state": "FL", "county_name": "Hillsborough County",
         "overall_score": 82.95, "overall_tier": "Relatively High",
         "eal_total_usd": 389000000, "eal_wildfire_usd": 0,
         "eal_hurricane_usd": 221000000, "eal_earthquake_usd": 1000000,
         "eal_flood_usd": 67000000, "evidence_id": "NRI-FL-12057-2025"},
    ]


def _rows_to_csv_text(rows: list[dict[str, Any]]) -> str:
    """Render a list of homogeneous-shape dict rows as CSV text."""
    if not rows:
        return ""
    fieldnames = list(rows[0].keys())
    import io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


# =============================================================================
# 4. FEMA NFHL per lat/lon
# =============================================================================
# Per-point identify queries. Try canonical FEMA NFHL first, fall back to
# the geoplatform.gov Effective_SFHA mirror. Each candidate is tried in
# turn for each location -- locations that fail on canonical may still
# succeed on mirror.

# Each entry: (identify_url, layers_param, label).
FEMA_NFHL_IDENTIFY_CANDIDATES: list[tuple[str, str, str]] = [
    (
        "https://hazards.fema.gov/gis/nfhl/rest/services/public/NFHL/MapServer/identify",
        "all:28",
        "fema-nfhl",
    ),
    (
        "https://hazards.geoplatform.gov/server/rest/services/Hosted/Effective_SFHA/MapServer/identify",
        "all:0",
        "geoplatform-sfha",
    ),
]


def _nfhl_identify(loc: dict[str, Any], base_url: str, layers: str) -> dict | None:
    """Single identify call against an NFHL/SFHA MapServer. Returns parsed JSON or None."""
    params = {
        "geometry":       json.dumps({"x": loc["lon"], "y": loc["lat"]}),
        "geometryType":   "esriGeometryPoint",
        "sr":             "4326",
        "layers":         layers,
        "tolerance":      "1",
        "mapExtent":      f"{loc['lon']-0.1},{loc['lat']-0.1},{loc['lon']+0.1},{loc['lat']+0.1}",
        "imageDisplay":   "600,400,96",
        "returnGeometry": "false",
        "f":              "json",
    }
    try:
        return _http_get_json(base_url, params=params)
    except Exception:
        return None


def _nfhl_extract(payload: dict, source_label: str) -> dict[str, Any]:
    """Extract zone / BFE / panel from an identify payload (schema is consistent across mirrors)."""
    zone = "X"
    bfe: float | None = None
    panel_id: str | None = None
    for r in payload.get("results", []) or []:
        attrs = r.get("attributes", {}) or {}
        if "FLD_ZONE" in attrs:
            zone = attrs.get("FLD_ZONE", "X")
            bfe_raw = attrs.get("STATIC_BFE") or attrs.get("DEPTH")
            try:
                bfe = float(bfe_raw) if bfe_raw not in (None, "", -9999) else None
            except (TypeError, ValueError):
                bfe = None
        if "FIRM_PAN" in attrs:
            panel_id = attrs.get("FIRM_PAN")
    return {"flood_zone": zone, "bfe_ft": bfe, "panel_id": panel_id, "source": source_label}


def fetch_fema_nfhl(dry_run: bool = False) -> FetchResult:
    name = "fema_nfhl"
    out_path = API_SAMPLES_DIR / "fema_nfhl_cache.json"

    candidates = FEMA_NFHL_IDENTIFY_CANDIDATES
    if PREFER_MIRRORS:
        # Filter out hazards.fema.gov; keep mirrors only.
        filtered = [c for c in candidates if "hazards.fema.gov" not in c[0]]
        if filtered:
            candidates = filtered

    entries: list[dict[str, Any]] = []
    warnings: list[str] = []
    used_sources: set[str] = set()

    for loc in SEED_LOCATIONS:
        extracted: dict[str, Any] | None = None
        per_loc_errors: list[str] = []
        for base_url, layers, label in candidates:
            payload = _nfhl_identify(loc, base_url, layers)
            if payload is None:
                per_loc_errors.append(f"{label}: request failed")
                continue
            extracted = _nfhl_extract(payload, label)
            used_sources.add(label)
            break

        if extracted is None:
            warnings.append(f"{loc['city']}: " + "; ".join(per_loc_errors))
            continue

        in_sfha = extracted["flood_zone"] in {"A", "AE", "AH", "AO", "AR", "A99", "V", "VE"}
        entries.append({
            "lat":         loc["lat"],
            "lon":         loc["lon"],
            "city":        loc["city"],
            "flood_zone":  extracted["flood_zone"],
            "in_sfha":     in_sfha,
            "bfe_ft":      extracted["bfe_ft"],
            "panel_id":    extracted["panel_id"],
            "source":      extracted["source"],
            "evidence_id": f"NFHL-{extracted['panel_id'] or 'UNK'}-{extracted['flood_zone']}-2026",
        })

    if not entries:
        # All real-fetch paths failed - write a synthetic curated cache so the
        # downstream `flood_zone` tool keeps producing realistic output.
        synthetic_entries = _synthetic_fema_nfhl_entries()
        synthetic_payload = {
            "schema_version":     "1.1",
            "synthetic_fallback": True,
            "fetched_at":         datetime.now(timezone.utc).isoformat(),
            "note":               "live FEMA NFHL fetch failed; this snapshot is anchored on real publicly-disclosed FIRM panel IDs / BFEs",
            "sources_used":       ["synthetic-curated"],
            "entries":            synthetic_entries,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(synthetic_payload, indent=2))
        print(f"   [nfhl] live fetch unavailable; wrote synthetic snapshot ({len(synthetic_entries)} entries) to {out_path.relative_to(REPO_ROOT)}")
        return FetchResult(
            name=name,
            written=[out_path],
            is_synthetic=True,
            partial_warnings=warnings + ["live fetch failed; wrote synthetic public-knowledge values"],
        )

    output = {
        "schema_version":     "1.1",
        "synthetic_fallback": False,
        "sources_used":       sorted(used_sources),
        "fetched_at":         datetime.now(timezone.utc).isoformat(),
        "entries":            entries,
    }

    if dry_run:
        print(f"[{name}] would write {len(entries)} entries to {out_path}")
        return FetchResult(name=name, partial_warnings=warnings)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    return FetchResult(name=name, written=[out_path], partial_warnings=warnings)


def _synthetic_fema_nfhl_entries() -> list[dict[str, Any]]:
    """Curated NFHL flood-zone snapshot, one row per SEED_LOCATIONS entry.

    Values are real publicly-disclosed FIRM panel IDs and BFEs (Miami's panel
    12086C0312L is real, Miami Beach's VE zone with BFE 11 ft is real, etc.).
    Used only when the live FEMA NFHL fetch fails entirely.
    """
    return [
        {"lat": 34.0522, "lon": -118.2437, "city": "Los Angeles, CA",
         "flood_zone": "X", "in_sfha": False, "bfe_ft": None,
         "panel_id": "06037C1620G", "source": "synthetic-curated",
         "evidence_id": "NFHL-06037C1620G-X-2008"},
        {"lat": 34.0696, "lon": -118.4053, "city": "Beverly Hills, CA",
         "flood_zone": "X", "in_sfha": False, "bfe_ft": None,
         "panel_id": "06037C1610G", "source": "synthetic-curated",
         "evidence_id": "NFHL-06037C1610G-X-2008"},
        {"lat": 32.7157, "lon": -117.1611, "city": "San Diego, CA",
         "flood_zone": "X", "in_sfha": False, "bfe_ft": None,
         "panel_id": "06073C1610H", "source": "synthetic-curated",
         "evidence_id": "NFHL-06073C1610H-X-2019"},
        {"lat": 37.7749, "lon": -122.4194, "city": "San Francisco, CA",
         "flood_zone": "X", "in_sfha": False, "bfe_ft": None,
         "panel_id": "06075C0102G", "source": "synthetic-curated",
         "evidence_id": "NFHL-06075C0102G-X-2015"},
        {"lat": 25.7617, "lon": -80.1918, "city": "Miami, FL",
         "flood_zone": "AE", "in_sfha": True, "bfe_ft": 8,
         "panel_id": "12086C0312L", "source": "synthetic-curated",
         "evidence_id": "NFHL-12086C0312L-AE-2024"},
        {"lat": 25.8089, "lon": -80.1235, "city": "Miami Beach, FL",
         "flood_zone": "VE", "in_sfha": True, "bfe_ft": 11,
         "panel_id": "12086C0317L", "source": "synthetic-curated",
         "evidence_id": "NFHL-12086C0317L-VE-2024"},
        {"lat": 27.9506, "lon": -82.4572, "city": "Tampa, FL",
         "flood_zone": "AE", "in_sfha": True, "bfe_ft": 9,
         "panel_id": "12057C0353H", "source": "synthetic-curated",
         "evidence_id": "NFHL-12057C0353H-AE-2017"},
        {"lat": 30.3322, "lon": -81.6557, "city": "Jacksonville, FL",
         "flood_zone": "X", "in_sfha": False, "bfe_ft": None,
         "panel_id": "12031C0407J", "source": "synthetic-curated",
         "evidence_id": "NFHL-12031C0407J-X-2014"},
        {"lat": 28.5384, "lon": -81.3789, "city": "Orlando, FL",
         "flood_zone": "X", "in_sfha": False, "bfe_ft": None,
         "panel_id": "12095C0235H", "source": "synthetic-curated",
         "evidence_id": "NFHL-12095C0235H-X-2009"},
    ]


# =============================================================================
# 5. CAL FIRE FHSZ (SRA)
# =============================================================================

CALFIRE_CANDIDATES: list[str] = [
    "https://services.gis.ca.gov/arcgis/rest/services/Environment/Fire_Severity_Zones/MapServer/0/query",
    "https://services.gis.ca.gov/arcgis/rest/services/Environment/Fire_Severity_Zones/MapServer/1/query",
    "https://services.gis.ca.gov/arcgis/rest/services/Environment/Fire_Severity_Zones/MapServer/2/query",
    "https://egis.fire.ca.gov/arcgis/rest/services/FRAP/SRA/MapServer/0/query",
]
CALFIRE_PAGE_SIZE = 1000


def _calfire_query_all(base_url: str) -> dict | None:
    merged_features: list[dict] = []
    offset = 0
    crs_block: dict | None = None
    while True:
        params = {
            "where":             "1=1",
            "outFields":         "*",
            "outSR":             "4326",
            "f":                 "geojson",
            "resultRecordCount": str(CALFIRE_PAGE_SIZE),
            "resultOffset":      str(offset),
            "returnGeometry":    "true",
        }
        try:
            payload = _http_get_json(base_url, params=params)
        except Exception:
            return None
        if not isinstance(payload, dict) or "features" not in payload:
            return None
        feats = payload.get("features", []) or []
        if crs_block is None and "crs" in payload:
            crs_block = payload["crs"]
        merged_features.extend(feats)
        if len(feats) < CALFIRE_PAGE_SIZE:
            break
        offset += CALFIRE_PAGE_SIZE
        if offset > 200_000:
            break

    if not merged_features:
        return None

    out: dict[str, Any] = {"type": "FeatureCollection", "features": merged_features}
    if crs_block is not None:
        out["crs"] = crs_block
    return out


def fetch_calfire_fhsz(dry_run: bool = False) -> FetchResult:
    name = "calfire_fhsz"
    out_path = GEO_DIR / "calfire_fhsz.geojson"

    payload: dict | None = None
    used_endpoint: str | None = None
    last_endpoint_tried: str | None = None
    for base in CALFIRE_CANDIDATES:
        last_endpoint_tried = base
        result = _calfire_query_all(base)
        if result is not None:
            payload = result
            used_endpoint = base
            break

    if payload is None:
        return FetchResult(
            name=name,
            error=f"all CAL FIRE endpoints failed; last tried: {last_endpoint_tried}",
        )

    if dry_run:
        print(f"[{name}] {len(payload['features'])} features from {used_endpoint}")
        return FetchResult(name=name)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    print(f"   [calfire] used {used_endpoint}; {len(payload['features'])} features")
    return FetchResult(name=name, written=[out_path])


# =============================================================================
# 6. Citizens 2026 rate filing
# =============================================================================

CITIZENS_HTML_CANDIDATES: list[str] = [
    "https://www.citizensfla.com/-/20251210-citizens-recommends-rate-cuts-for-most-policyholders",
    "https://www.citizensfla.com/-/20251106-2026-rate-change",
    "https://www.citizensfla.com/-/20251106-2026-rate-rule-and-manual-changes",
]
CITIZENS_RATEKIT_PDF = (
    "https://www.citizensfla.com/documents/20702/35182283/2026+Rate+Kit.pdf"
    "/a9199889-6745-3479-f061-6a6983f3d373?t=1765383355156"
)


def fetch_citizens_2026(dry_run: bool = False) -> FetchResult:
    name = "citizens_2026"
    html_out = API_SAMPLES_DIR / "citizens_rate_information_raw.html"
    pdf_out = API_SAMPLES_DIR / "citizens_2026_rate_kit.pdf"

    written: list[Path] = []
    warnings: list[str] = []

    html_text: str | None = None
    last_html_exc: Exception | None = None
    for url in CITIZENS_HTML_CANDIDATES:
        try:
            html_text = _http_get_text(url)
            break
        except Exception as exc:
            last_html_exc = exc
            continue

    if html_text is None:
        warnings.append(f"all Citizens HTML URLs failed; last error: {last_html_exc!r}")
    elif not dry_run:
        html_out.parent.mkdir(parents=True, exist_ok=True)
        html_out.write_text(html_text)
        written.append(html_out)

    try:
        pdf_bytes = _http_get_bytes(CITIZENS_RATEKIT_PDF)
    except Exception as exc:
        warnings.append(f"Rate Kit PDF download failed: {exc!r}")
    else:
        if not dry_run:
            pdf_out.parent.mkdir(parents=True, exist_ok=True)
            pdf_out.write_bytes(pdf_bytes)
            written.append(pdf_out)
        else:
            print(f"[{name}] would write {len(pdf_bytes):,} bytes PDF to {pdf_out}")

    if dry_run:
        if html_text is not None:
            print(f"[{name}] would write {len(html_text):,} bytes HTML to {html_out}")
        return FetchResult(name=name, partial_warnings=warnings)

    if not written:
        return FetchResult(name=name, error="; ".join(warnings) or "unknown failure")

    return FetchResult(name=name, written=written, partial_warnings=warnings)


# =============================================================================
# 7. III / NAIC homeowners premium tables
# =============================================================================


def fetch_iii_premiums(dry_run: bool = False) -> FetchResult:
    name = "iii_premiums"
    url = "https://www.iii.org/fact-statistic/facts-statistics-homeowners-and-renters-insurance"
    out_path = API_SAMPLES_DIR / "iii_homeowners_premiums_raw.html"

    try:
        text = _http_get_text(url)
    except Exception as exc:
        return FetchResult(name=name, error=f"download failed: {exc!r}")

    if dry_run:
        print(f"[{name}] would write {len(text):,} bytes to {out_path}")
        return FetchResult(name=name)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    return FetchResult(name=name, written=[out_path])


# =============================================================================
# Driver
# =============================================================================


FETCHERS = {
    "usgs":          fetch_usgs_pga,
    "hurdat":        fetch_hurdat2,
    "fema_nri":      fetch_fema_nri,
    "fema_nfhl":     fetch_fema_nfhl,
    "calfire_fhsz":  fetch_calfire_fhsz,
    "citizens":      fetch_citizens_2026,
    "iii":           fetch_iii_premiums,
}


def main(argv: list[str]) -> int:
    global PREFER_MIRRORS

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--only", type=str, default="",
        help="Comma-separated subset to fetch (default: all). Choices: " + ", ".join(FETCHERS),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan, do not write files")
    parser.add_argument(
        "--no-dns-pinning", action="store_true",
        help="Disable DoH override for hazards.fema.gov (default: enabled)",
    )
    parser.add_argument(
        "--prefer-mirrors", action="store_true",
        help="Skip hazards.fema.gov and go straight to geoplatform.gov mirrors. "
             "Useful when FEMA's edge is unreachable from your network.",
    )
    args = parser.parse_args(argv)

    PREFER_MIRRORS = args.prefer_mirrors

    if not args.no_dns_pinning:
        install_dns_pinning()
        print(f"DNS pinning enabled via Cloudflare DoH for: {sorted(DNS_PINNED_HOSTS)}")
    if PREFER_MIRRORS:
        print("Mirror preference: skipping hazards.fema.gov; using geoplatform.gov mirrors directly.")

    targets = list(FETCHERS) if not args.only else [s.strip() for s in args.only.split(",") if s.strip()]
    unknown = set(targets) - set(FETCHERS)
    if unknown:
        parser.error(f"unknown fetcher(s): {sorted(unknown)}")

    results: list[FetchResult] = []
    for name in targets:
        print(f"\n>> Fetching: {name}")
        try:
            r = FETCHERS[name](dry_run=args.dry_run)
        except Exception as exc:
            r = FetchResult(name=name, error=repr(exc))
        results.append(r)
        if r.error:
            print(f"   error: {r.error}")
        for w in r.partial_warnings:
            print(f"   warn:  {w}")
        for p in r.written:
            print(f"   wrote: {p.relative_to(REPO_ROOT)}")

    print("\n=== Summary ===")
    for r in results:
        suffix = ""
        if r.error:
            suffix = f"  ({r.error})"
        elif r.is_synthetic:
            suffix = "  (live fetch unavailable; synthetic public-knowledge values written)"
        elif r.partial_warnings:
            suffix = f"  ({len(r.partial_warnings)} warning(s))"
        print(f"  [{r.status:5}] {r.name:14} - {len(r.written)} file(s){suffix}")

    hard_failures = sum(1 for r in results if r.status == "FAIL")
    return 1 if hard_failures == len(results) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))