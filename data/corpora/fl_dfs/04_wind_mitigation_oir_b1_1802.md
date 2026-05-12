---
corpus: fl_dfs
jurisdiction: FL
evidence_id: FLDFS-WIND-MIT-01
source_url: https://www.floir.com/Office/InspectionForms.aspx
title: Florida Wind Mitigation - OIR Form OIR-B1-1802
---

# Florida wind-mitigation discounts under OIR-B1-1802

Florida law requires every personal residential property insurer to apply mitigation premium credits when an applicant or insured documents qualifying wind-resistive construction features. The vehicle for documentation is **Office of Insurance Regulation Form OIR-B1-1802** (the "Uniform Mitigation Verification Inspection Form"), completed by a qualified inspector. The combined credit can reach approximately **45% of the wind portion of premium** for a fully-mitigated structure.

**What the rule requires.** Insurers must (1) accept a properly completed OIR-B1-1802 form from any licensed building inspector, contractor, engineer, or architect listed in §627.711, Fla. Stat.; (2) apply the corresponding mitigation discounts to the wind-loss component of the rate filing; and (3) re-rate the policy at renewal when an updated form is submitted documenting new mitigation features. The form documents seven categories: roof covering, roof deck attachment, roof-to-wall connection, roof geometry, secondary water resistance, opening protection, and gable-end bracing.

**Trigger conditions in plain words.** Whenever the customer profile indicates `state == "FL"`, the rule applies — every Florida residential policy must offer the inspection mechanism and apply discounts when the form is submitted. The trigger does not depend on whether the customer has *yet* submitted a form; the offer itself is the obligation.

**Action the rule requires.** Add a `required_offer` to the StatutoryEngineOutput with `offer == "wind_mitigation_inspection"` and `form == "OIR-B1-1802"`. The advisory severity reflects that the discount is voluntary on the customer's part (they choose whether to commission the inspection) but the *offer* itself is mandatory.

**Edge cases and related provisions.** §627.711 lists the qualifying inspector credentials. §627.711(1)(a) prohibits insurers from refusing to accept a form solely because the inspector is not affiliated with the insurer. For homes built after the 2002 Florida Building Code, certain mitigation credits are presumed and the form may not be required to claim them. Newly constructed homes (post-2002) typically receive a baseline credit of ~20% even without OIR-B1-1802 submission.
