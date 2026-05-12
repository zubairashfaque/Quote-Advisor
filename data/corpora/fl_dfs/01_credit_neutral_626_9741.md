---
corpus: fl_dfs
jurisdiction: FL
evidence_id: FLDFS-CREDIT-01
source_url: https://www.myfloridacfo.com/division/consumers
title: Florida DFS - Credit-Based Insurance Scoring (Fla. Stat. §626.9741)
---

# Florida credit-based insurance scoring - the §626.9741 framework

Florida regulates the use of credit information in personal lines underwriting and rating under Fla. Stat. §626.9741, codified through Florida Department of Financial Services guidance. Two provisions matter most for homeowners pricing: (1) the neutral-treatment rule when credit information is unavailable, and (2) the prohibition on using certain factors directly within the credit score.

**Neutral treatment when credit is unavailable.** Fla. Stat. §626.9741(7) requires that when an insurer uses credit information as a rating factor and credit information for an applicant or insured is not available, the insurer must apply a neutral credit factor. In practice this is operationalized as a 1.0× multiplier on the credit-banded portion of the rate calculation. The insurer cannot penalize the applicant or default to a "worst-case" credit band.

This is the load-bearing protection for Profile B in this system: a Florida applicant with `credit_score: null` is treated identically to one with a 700-720 credit score for the credit-multiplier component of pricing.

**What "unavailable" means.** Per §626.9741(8), an insurer must order credit information at new business and at every other renewal (i.e., every 24 months). If the credit bureau returns "no hit" or "thin file" responses, the insurer must apply the neutral factor; it cannot decline to write or use a different factor in lieu. The CreditScoreImputerTool in this system encodes this rule deterministically.

**Prohibited inputs to the credit score itself.** §626.9741(2) prohibits the use of any of the following as a primary factor within the credit-based score itself: income, gender, address, ZIP code, ethnic group, religion, marital status, nationality. While these factors might still appear in a vendor's proprietary score, the insurer must demonstrate they are not used in a way that produces a primary disparate-impact outcome.

**Notice requirements.** Insurers using credit information must provide adverse-action notice to any consumer whose rate is increased due to credit; this aligns with FCRA notice requirements but is enforced separately by FL DFS. Mortgage applicants are entitled to request a re-rating when credit information improves materially.

**Compliance Officer Council persona.** In this system's 4-persona Council, the Compliance Officer holds VETO power over any output that fails to apply the §626.9741(7) neutral treatment when credit is null. The StatutoryRulesEngine fires FL-CREDIT-NEUTRAL upstream as the first-line defense; the Council is the second backstop.
