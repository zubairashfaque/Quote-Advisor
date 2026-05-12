---
corpus: gse_lender
jurisdiction: national
evidence_id: GSE-PROP-INS-01
source_url: https://selling-guide.fanniemae.com/sel/b7-3-02/property-insurance-requirements-one-four-unit-properties
title: Fannie Mae Selling Guide B7-3-02 - Property Insurance Requirements
---

# GSE property-insurance floor for federally-backed mortgages

Fannie Mae's Selling Guide section **B7-3-02** establishes minimum property-insurance requirements for any one-to-four-unit residential property securing a Fannie Mae-eligible loan. Freddie Mac's parallel requirement (Single-Family Seller/Servicer Guide §4703.2) imposes substantively identical limits. Together the two GSEs underwrite or guarantee the majority of US conforming mortgages, so this floor effectively binds for any homeowner with a conforming loan.

**What the rule requires.** Dwelling coverage (Coverage A on a standard HO-3 or HO-5 form) must be **at least the lesser of**: (1) the property's full replacement cost value (RCV), or (2) the unpaid principal balance (UPB) of the mortgage, *provided the second figure is enough to repair or replace the dwelling*. In practice this means the lender will not accept a policy with Coverage A below the smaller of RCV and UPB. Settlement basis must be **replacement cost**, not actual cash value (ACV) — ACV settlement is unacceptable for the dwelling under B7-3-02.

**Trigger conditions in plain words.** Whenever the customer profile indicates `has_mortgage == true` (regardless of state), the rule fires. The property does not need to be in any particular hazard zone; the requirement is universal across GSE-eligible mortgages. If `has_mortgage` is unknown or false, the rule does not fire.

**Action the rule requires.** Set a `floor` entry in the StatutoryEngineOutput keyed on `coverage_a` with `value_rule == "min(replacement_cost, unpaid_principal_balance)"`. The severity is `mandatory` because the lender will not close the loan or will force-place insurance if the floor is not met.

**Edge cases and related provisions.** Condos and PUDs follow a different path (B7-3-03 and B7-3-04 respectively, master-policy frameworks). Loss-payable clauses must name the lender as mortgagee. The 60-day-notice-of-cancellation requirement to the lender is separate but related. For NFIP flood policies in SFHA properties, the GSE floor stacks with the NFIP-MANDATORY rule; the customer effectively needs both policies.
