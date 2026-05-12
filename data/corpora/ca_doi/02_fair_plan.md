---
corpus: ca_doi
jurisdiction: CA
evidence_id: CDI-FAIR-01
source_url: https://www.cfpnet.com/about/
title: California DOI - FAIR Plan and DIC Wrap
---

# California FAIR Plan + Difference-In-Conditions wrap

The California FAIR Plan (Cal. Ins. Code §10090 et seq.) is the state's last-resort property insurance market for owners who cannot obtain coverage in the voluntary admitted market. Created in 1968 after the 1965 Watts riots and expanded after 1990s wildfires, it is operated by a syndicate of admitted carriers under CDI oversight.

**What FAIR Plan covers.** A "basic peril" residential dwelling policy: fire, lightning, internal explosion, smoke. Coverage A only - NO Coverage B, C, D, E, or F. The form is intentionally narrow because the FAIR Plan is meant to be a market of last resort, not a substitute for an admitted HO-3.

**Difference-In-Conditions (DIC) wrap.** Because FAIR Plan excludes the perils most homeowners care about (theft, water damage, liability, ALE), insureds buy a separate Difference-In-Conditions companion policy from a non-admitted (surplus lines) or specialist admitted carrier. The DIC wrap fills the gaps - personal property, liability, theft, ALE, water damage, and increased Coverage A above the FAIR Plan limit.

**When FAIR + DIC is the route.** Triggered when the property sits in CAL FIRE Fire Hazard Severity Zone (FHSZ) High or Very High and the admitted market declines. The Eligibility Triage agent's Tree-of-Thoughts pruning typically scores the FAIR + DIC branch highest in such cases. The combined cost is generally 1.5-2.5x what an admitted HO-3 would have charged, plus the administrative overhead of two policies.

**Limits, surcharges, and recent reforms.** FAIR Plan dwelling limits historically capped at $1.5M for residential; raised to $3M for the dwelling and $1M for personal property in 2024 under the Sustainable Insurance Strategy. A Cumulative Loss Surcharge applies after major catastrophe years; insureds and admitted carriers share the surcharge per the assessment formula.

**FAIR Plan eligibility check.** A property is eligible if it has been declined by at least one admitted carrier OR sits in a designated brush/wildfire zone with no admitted appetite. The CAFairPlanEligibilityTool encodes the FHSZ × ZIP3 matrix the Risk Assessment agent uses to predict eligibility.

**Renewal vs. shopping back.** Insureds on FAIR + DIC should re-shop the admitted market each renewal as Sustainable Insurance Strategy 2024 brings carriers back into wildfire-distressed ZIPs. The Follow-up Explanation agent surfaces this advisory when the customer asks "is this the cheapest I can get?"
