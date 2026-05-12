---
corpus: iii_handbook
jurisdiction: national
evidence_id: III-HB-ACTUARIAL-01
source_url: https://www.iii.org/publications/insurance-handbook
title: III Handbook - Actuarial Methods for Homeowners
---

# Actuarial methods - how rates are actually set

Underneath the multiplier chain is a body of actuarial method: pure premium projection, loss-cost trending, catastrophe-loaded ratemaking, and reinsurance-cost recovery. The III Handbook describes these in narrative form; this system's Pricing Agent uses the multiplier chain as a forward-look proxy.

**Pure premium = frequency × severity.** The actuarial first-principles approach: estimate claim frequency (claims per 1,000 insured years) for each peril, multiply by expected severity (average claim size), sum across perils. Add expense load and profit provision to get the indicated premium. The III handbook's reference: 2024 frequency for owner-occupied SFD nationally ran ~5.8 claims per 100 insured-years for property losses; mean severity ran ~$15,500 (skewed by a long CAT tail).

**Loss-cost trending.** Insurance is forward-looking: the rate effective in 2026 must cover losses in 2026-2027. Loss-cost trending applies an annual factor (typically 6-12% in 2024-2026 due to construction-cost inflation, labor scarcity, and CAT activity) to historical loss data. Under-trending is the load-bearing cause of inadequate rates.

**Catastrophe loading.** A long-tail provision for low-frequency, high-severity events (hurricanes, wildfires, earthquakes, severe convective storms). Modern catastrophe models (RMS, AIR, Karen Clark, Verisk Touchstone) simulate millions of synthetic event years and produce annual average loss + standard deviation per zip / county. The CAT load is typically 20-50% of total premium in CAT-exposed regions.

**Reinsurance pass-through.** Insurers cede a portion of catastrophe risk to reinsurers; the cost of reinsurance is loaded into the gross direct premium. Reinsurance market hardening 2022-2025 (driven by Hurricane Ian 2022 and CA wildfire 2023-24 insolvencies) added 15-30% to reinsurance costs, which flowed through to direct rates over 12-24 months. CA's 2024 Sustainable Insurance Strategy formally permits net-of-reinsurance ratemaking - a major change.

**Investment income offset.** Insurers earn investment yield on reserves; rising 10-year Treasury yields 2022-2024 mean rate filings can incorporate higher investment-income offsets, partially offsetting loss-cost increases. The CDI's actuarial review tests for over-loading on investment income (rates can't be too low because of conservative investment assumptions).

**Why this matters for this system.** The Pricing Agent uses a multiplier chain as a defensible heuristic, NOT a true ratemaking calculation. The README assumptions section and DEC-0004 are explicit about this trade-off. A production deployment would replace the curated multipliers JSON with an actuarial loss-cost model; the architecture (Pricing as ReWOO with deterministic Workers + LLM Solver) is designed to absorb that swap without changing the orchestration.
