---
corpus: ca_doi
jurisdiction: CA
evidence_id: CDI-PROP103-01
source_url: https://www.insurance.ca.gov/0250-insurers/0300-insurers/0200-bulletins/bulletin-notices-commiss-opinion/index.cfm
title: California DOI - Proposition 103 Overview
---

# Proposition 103 - the load-bearing California rate-regulation regime

Proposition 103 (1988) is the foundational California personal-lines rate-regulation framework. Three provisions matter most for homeowners pricing: (1) the prior-approval rate-filing requirement, (2) mandatory rating factors with prohibited-factor exclusions, and (3) the California Earthquake Authority offer requirement.

**Prior-approval rates.** Cal. Ins. Code §1861.05 requires insurers to file proposed rates with the CDI and obtain approval before use. Filings must be supported by actuarial data showing the rate is not excessive, inadequate, or unfairly discriminatory. The Department of Insurance reviews and may set a hearing on consumer challenges; rate changes can take 12-24 months from filing to effective.

**Mandatory and prohibited rating factors.** Cal. Ins. Code §1861.02 designates the only factors that may be used as PRIMARY rating bases: in personal auto, driving record, miles driven, and years of driving experience. For homeowners, the regulation is somewhat looser, but several factors are categorically prohibited as primary:

- **Credit-based insurance scores** are prohibited (Cal. Code Regs. tit. 10 §2632.5). The Coverage and Pricing agents must drop credit_score from their context entirely.
- **Age, gender, marital status, ethnicity, religion, national origin** cannot be used as primary rating factors.
- Properties cannot be rated solely on ZIP code (the "ZIP code rule" - Cal. Ins. Code §1861.02(a) extended to homeowners by Sustainable Insurance Strategy 2024).

**Sustainable Insurance Strategy (2024).** Commissioner Lara's December 2023 - January 2024 framework allowing the use of catastrophe modeling (forward-looking CAT models) and net-of-reinsurance ratemaking in exchange for required writings in distressed wildfire markets. Marks the first significant CAT-modeling permission in CA personal lines.

**Earthquake Coverage offer.** Cal. Ins. Code §10081 requires every residential property insurer in California to offer earthquake coverage at every new business and renewal. This is typically satisfied by the California Earthquake Authority Companion Policy program, which the insurer markets but does not underwrite.

The Compliance Officer Council persona retains VETO power on any output that violates Prop 103 - the StatutoryRulesEngine drops credit_score before any LLM sees the profile, but the Council acts as a second backstop.
