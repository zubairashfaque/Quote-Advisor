---
corpus: ca_doi
jurisdiction: CA
evidence_id: CDI-STDFORM-01
source_url: https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=INS&sectionNum=2071
title: California Insurance Code §2071 - Standard Fire Policy Form
---

# California §2071 — the standard fire policy form requirement

California Insurance Code §2071 prescribes the **standard fire policy form** that every insurer writing residential property coverage in California must use as the baseline for fire-peril insuring agreements. The form establishes uniform definitions, exclusions, conditions, and the proof-of-loss process; carriers may add coverage by endorsement but cannot subtract from the §2071 baseline.

**What the rule requires.** Any homeowners or dwelling fire policy issued in California must incorporate the §2071 standard form language for the fire-peril insuring agreement, the conditions of coverage (including the 60-day proof-of-loss window), the loss-payment timeline, and the appraisal-and-arbitration clause. The Department of Insurance has published the canonical text of the form; insurers' policy language for these sections must be substantively identical, even if the surrounding HO-3 / HO-5 text varies by carrier.

**Trigger conditions in plain words.** Whenever the customer profile indicates `state == "CA"` and the product line is residential property (which is always true for this system), the rule fires. There is no profile-level exception — every California homeowners quote carries the §2071 form requirement.

**Action the rule requires.** Add `"CA-Standard-Form-Fire"` to the `required_forms` list in the StatutoryEngineOutput. Downstream this surfaces in the QuoteOutput as a form id the policy paperwork must include. The rule has `severity: "mandatory"` — failure to incorporate the form invalidates the policy for §2071 compliance purposes.

**Edge cases and related provisions.** §2070 establishes the legislative basis for the standard-form mandate; §2080 governs the same form's application to surplus-lines residential coverage. The form is silent on perils other than fire (e.g., wind, water) — those are governed by separate statutory and contractual provisions. For DIC (Difference-In-Conditions) wraps over a FAIR Plan policy, the §2071 form applies to the FAIR Plan layer; the DIC carrier need not duplicate it.
