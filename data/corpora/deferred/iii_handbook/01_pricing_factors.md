---
corpus: iii_handbook
jurisdiction: national
evidence_id: III-HB-PRICING-01
source_url: https://www.iii.org/publications/insurance-handbook
title: III Handbook - Homeowners Pricing Factors
---

# Homeowners pricing - the major rating factors

The Insurance Information Institute's Insurance Handbook describes the canonical structure of a homeowners base premium. The architecture is consistent across most US carriers: a state base premium scaled by dwelling value, then multiplied by a chain of factors representing peril exposure, insured characteristics, and discount stacking.

**The base premium.** Each state has a published statewide median (or HO-3 reference) for a $250K dwelling. This is the rating ANCHOR; all subsequent multipliers operate against it. Sources include NAIC's annual Homeowners Insurance Report, Bankrate / III state averages, and per-carrier rate filings on the state DOI website.

**Dwelling-value scaling.** Premium scales nonlinearly with Coverage A. The standard rule of thumb: each $100K of dwelling above $250K adds about 18% to base. So a $900K home rates approximately 2.2x the $250K base ($900K - $250K = $650K excess; 6.5 × 0.18 = +1.17; total = 2.17x).

**Peril multipliers.** A multiplicative chain by peril and severity tier. Typical magnitudes:
- Wildfire: Low 1.0× / Moderate 1.3× / High 1.7× / Very High 2.0× of base.
- Hurricane: Low 1.0× / Moderate 1.4× / High 1.8× / Very High 2.2× of base.
- Seismic: Low 1.0× / Moderate 1.05× / High 1.12× / Very High 1.20× of base. (Lower multipliers because EQ is typically a separate companion policy.)
- Flood: zone X 1.0× / AE 1.5× / VE 2.0× (NFIP and private flood pricing differ; this is a rough proxy).

**Insured-characteristic multipliers.**
- Claims history: 0 prior 1.0× / 1 prior 1.33× / 2 prior 1.7× / 3+ prior 2.5×.
- Pool: 1.12× (attractive nuisance liability uplift).
- Restricted dog breed: 1.10-1.50× depending on breed.
- Credit-banded factor: 0.90× excellent / 0.95× good / 1.10× fair / 1.30× poor. Subject to state law (CA prohibits, FL requires neutral when unavailable).

**Discount stack.**
- Multi-policy bundle: 10-25% off home (when paired with auto, life, umbrella).
- Wind mitigation (FL): up to 45% off the wind-exposed component per OIR-B1-1802.
- Defensible space (CA): up to 15% off when documented; carrier-specific.
- Claims-free / loyalty: 5-10% after 3+ years claim-free.

**Why the chain composition matters.** Multiplicative chains compound: a $1,976 CA base × 1.33 (1 claim) × 2.0 (Very High wildfire) × 1.12 (pool) × 3.0 (home value scaling) = $17,663 point estimate. The Pricing Agent's ReWOO Solver shows this as a transparent factor chain with each evidence_id cited; the Follow-up Agent's Self-Ask decomposition surfaces the largest multipliers when the customer asks "why is this expensive?".
