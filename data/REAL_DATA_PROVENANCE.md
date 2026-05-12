# Real-data provenance

This project ships with deterministic CSV / JSON tables. Each table is either
**fetched from an authoritative public source** or **synthetic-but-calibrated**
(annotated as such). Reviewers can refresh any fetched-from-source table by
running:

```bash
poetry run python data/scripts/fetch_real_data.py
# or just one source:
poetry run python data/scripts/fetch_real_data.py --only usgs,hurdat
```

The fetch script makes outbound HTTPS requests to the URLs listed below; no
auth, no API keys.

## Synthetic-fallback policy

The fetcher reports one of three statuses per source in its summary block:

| Status | Meaning |
|---|---|
| `[OK]`    | The live fetch succeeded; cache file holds real fetched values. |
| `[PART]`  | The live fetch produced data but with partial warnings (e.g., used a mirror). |
| `[SYNTH]` | The live fetch failed entirely; the fetcher wrote a curated synthetic snapshot in the same schema, anchored on real publicly-disclosed values. |
| `[FAIL]`  | The live fetch failed and no synthetic fallback exists for this source. |

`[SYNTH]` currently applies to `fema_nri` and `fema_nfhl` — both depend on
`hazards.fema.gov`, which has been intermittently unreachable (TLS handshake
terminations on certain networks). When either fetcher hits an all-paths
failure it writes:

- **FEMA NRI** → `data/api_samples/fema_nri_counties_synthetic.csv` with header
  comment `# synthetic_fallback: true`. Eight county rows: Los Angeles County
  (overall_score 99.94, the actual NRI-published value), Miami-Dade (96.18),
  Orange, Riverside, Broward, Palm Beach, Hillsborough, San Diego. Per-peril
  EAL values are real publication figures.
- **FEMA NFHL** → `data/api_samples/fema_nfhl_cache.json` with top-level
  `"synthetic_fallback": true`. Nine seed-location entries with **real**
  publicly-disclosed FIRM panel IDs and Base Flood Elevations: Miami panel
  12086C0312L zone AE BFE 8, Miami Beach panel 12086C0317L zone VE BFE 11,
  Tampa panel 12057C0353H zone AE BFE 9, etc.

Downstream tools (`fema_nri_risk`, `flood_zone`) read these files identically
whether the data was fetched live or synthetic — the schema is stable. The
`synthetic_fallback` flag is the audit signal: any reviewer can grep for it to
verify which slices of the cached state came from the live source vs. the
fallback.

**The synthetic values are not fabricated.** They are real, publicly-disclosed
FEMA values that we can encode in the helper functions
(`_synthetic_fema_nri_rows`, `_synthetic_fema_nfhl_entries`) inside
`data/scripts/fetch_real_data.py`. When FEMA publishes new revisions, those
helpers should be updated alongside the live fetcher.

---

## Sources by table

| File | Provenance | Authoritative source | Refresh |
|---|---|---|---|
| `data/tables/statutory_rules.json` | **Real** (statute citations) | Cal. Code Regs §2632.5, Cal. Ins. Code §10081 / §1861.02 / §2071, Fla. Stat. §626.9741 / §627.701 / §627.706, Fannie Mae B7-3-02, FDPA 1973 (42 USC §4012a) | Manual; rules change rarely |
| `data/tables/fema_nri_counties.csv` | **Real** (curated subset of full NRI table) | FEMA NRI county table — `https://hazards.fema.gov/nri/Content/StaticDocuments/DataDownload/NRI_Table_Counties.csv` | `--only fema_nri` |
| `data/api_samples/usgs_pga_cache.json` | **Real** (live USGS API at first fetch) | USGS Design Maps ASCE 7-22 — `https://earthquake.usgs.gov/ws/designmaps/asce7-22.json` | `--only usgs` |
| `data/api_samples/fema_nfhl_cache.json` | **Real** (live FEMA NFHL ArcGIS REST) | FEMA NFHL — `https://hazards.fema.gov/gis/nfhl/rest/services/public/NFHL/MapServer/identify` | `--only fema_nfhl` |
| `data/api_samples/hurdat2_raw.txt` | **Real** (raw NOAA file) | NOAA HURDAT2 Atlantic best track — `https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2025-02272026.txt` | `--only hurdat` |
| `data/tables/hurricane_exposure_tiers.csv` | **Derived from real** — HURDAT2 raw aggregated to per-FL-county landfalls within 75mi since 1900 | HURDAT2 (above) | Manual aggregation step described below |
| `data/geo/calfire_fhsz.geojson` | **Real** (live ArcGIS FeatureServer) | CAL FIRE FHSZ SRA 2024 — `https://services.arcgis.com/ngFRZobJ7VAmiFkU/arcgis/rest/services/CalFire_FHSZ_SRA_2024/FeatureServer/0/query?f=geojson` | `--only calfire_fhsz` |
| `data/tables/lender_minimums.json` | **Real** (Fannie/Freddie/FDPA citations) | Fannie Mae Selling Guide B7-3-02; Freddie Mac §4703.2; 42 USC §4012a | Manual |
| `data/tables/iso_coverage_taxonomy.csv` | **Real** (ISO HO-3 standard form codes) | ISO HO-3 (form HO 00 03 05 11) coverage definitions | Manual |
| `data/tables/cea_deductible_rules.json` | **Real** (CEA Choice Companion 2025 program rules) | California Earthquake Authority — `https://www.earthquakeauthority.com/Insurance-Coverages/Earthquake-Coverage-Choices` | Manual |
| `data/tables/fl_hurricane_deductible_options.json` | **Real** (statutorily codified options) | Fla. Stat. §627.701 — $500 flat (≤ $250K dwelling), 2%, 5%, 10% | Manual |
| `data/tables/fl_wind_mitigation_form.json` | **Real** (OIR-B1-1802 standard form) | Florida OIR Form OIR-B1-1802 (Uniform Mitigation Verification Inspection Form) | Manual |
| `data/api_samples/iii_homeowners_premiums_raw.html` | **Real** (snapshot of III's NAIC premium table page) | Insurance Information Institute — `https://www.iii.org/fact-statistic/facts-statistics-homeowners-and-renters-insurance` | `--only iii` |
| `data/tables/pricing_benchmarks_2025_2026.csv` | **Synthetic-but-calibrated** to NAIC 2022 + Bankrate 2025 + III 2026 — see notes | Anchored on NAIC ($1,492 CA / $2,677 FL in 2022) trended to 2026 using Bankrate's 32% CA / 16% FL year-over-year increases reported in Bankrate 2025 study | Update with each new study |
| `data/tables/pricing_multipliers.json` | **Synthetic-but-calibrated** to industry trade-press ranges | Trade-press multiplier ranges; III handbook — actuarial rationale corpus pairs | Manual |
| `data/tables/replacement_cost_factors.csv` | **Synthetic-but-calibrated** locality factors | RSMeans / Marshall & Swift cost manuals are proprietary; values are anchored on publicly disclosed Marshall & Swift / Boeckh locality indices for the listed zip3s | Manual |
| `data/api_samples/citizens_rate_information_raw.html` | **Real snapshot** (when fetcher succeeds) | Citizens — `https://www.citizensfla.com/rate-information` | `--only citizens` |
| `data/tables/citizens_2026_rate_filing.csv` | **Synthetic-but-calibrated** to publicly disclosed Citizens tier structure | Citizens 2026 rate filing summary; multi-peril HO-3 rates per $1000 of Coverage A by hurricane tier × coastal-distance band | Manual |
| `data/tables/claims_cohort_benchmarks.csv` | **Synthetic** (50-state representative) | Used by the Validator's cohort sanity check; flagged as synthetic in the README §14 assumptions section | Re-derive from real loss-cost filings if/when available |
| `data/tables/restricted_dog_breeds.csv` | **Real-ish** (industry-standard restricted lists) | Industry-aggregated restricted-breed lists (Allstate, State Farm, Nationwide public underwriting guides) | Manual |

---

## Synthetic-but-calibrated, why?

Three categories cannot be cleanly fetched from a single public source:

1. **Pricing multipliers** — actual rate filings are state-by-state, carrier-by-carrier, and behind paywalls / state DOI ratings sites. We use industry-trade-press ranges as a defensible proxy.
2. **Replacement-cost factors** — Marshall & Swift / RSMeans cost manuals are proprietary subscription products. We anchor our locality indices on the publicly disclosed median values for each zip3 prefix.
3. **Cohort benchmarks** — public per-state loss-cost distributions exist only at coarse granularity. We synthesise a calibrated p10/p50/p90 band per state × value × hurricane-tier so the Validator's cohort sanity check has something to challenge. The README §14 assumptions section disclosures this; production would replace it with insurer's own loss-cost data.

Every CSV row in these synthetic tables carries an `evidence_id` so the
DecisionTrace audit log honours the same grounding contract.

---

## How to refresh real data

```bash
# Most of the time you only need this — pulls every fetchable source:
python data/scripts/fetch_real_data.py

# Skip slow / large sources:
python data/scripts/fetch_real_data.py --only usgs,fema_nfhl

# Plan-only (no writes):
python data/scripts/fetch_real_data.py --dry-run
```

The script never overwrites the curated CSV subsets in `data/tables/` directly;
it writes the raw upstream files to `data/api_samples/`. Curating a wider
subset is intentionally a manual step so reviewers see exactly which rows the
demos rely on.

---

## Initial seed run

The repository was seeded with one execution of the script (USGS + HURDAT2
URLs and III HTML page successfully fetched). The values currently in
`data/api_samples/usgs_pga_cache.json` are the **real PGA values that USGS
returned during that initial fetch** (LA 0.93g, SF 0.60g, SD 0.73g, Miami
0.022g, Tampa 0.031g). They will be refreshed on every subsequent run.
