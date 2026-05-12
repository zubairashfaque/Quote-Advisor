"""Coverage Recommendation Agent (Plan-and-Execute).

Two LLM calls: planner emits an ordered 4-step plan, executor runs each step
deterministically (calling the right tool with the planner's inputs) and
assembles the RecommendedCoverage list.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .. import prompts
from ..decision_trace import append_node, make_node
from ..llm_registry import AgentRole, get_llm
from ..schemas import RecommendedCoverage
from ..tools import (
    cea_earthquake_recommender,
    coverage_rules,
    coverage_taxonomy,
    fl_hurricane_deductible,
    lender_floor,
    replacement_cost,
    wind_mitigation_discount,
)


class CoverageStep(BaseModel):
    step_number: int
    action: str
    tool_to_call: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)


class CoveragePlan(BaseModel):
    steps: list[CoverageStep]


def coverage_node(state: dict[str, Any]) -> dict[str, Any]:
    """Plan-then-execute coverage recommendation."""
    sanitized = state.get("sanitized_profile") or {}
    state_code = (sanitized.get("state") or "").upper()
    home_value = float(sanitized.get("home_value") or 0)
    has_mortgage = bool(sanitized.get("has_mortgage", True))  # assume mortgage by default
    in_sfha = bool(sanitized.get("in_sfha", False))

    # ---- Step 1: floors via lender + statutory --------------------------------
    rcv = replacement_cost.invoke({
        "state": state_code, "home_value_usd": home_value, "sqft": 2400.0,
    })
    floor = lender_floor.invoke({
        "has_mortgage": has_mortgage, "in_sfha": in_sfha,
        "home_value_usd": home_value, "rebuild_cost_usd": rcv["rebuild_cost_usd"],
    })
    rules = coverage_rules.invoke({
        "state": state_code, "home_value_usd": home_value,
        "has_mortgage": has_mortgage, "in_sfha": in_sfha,
    })

    cov_a = max(floor["coverage_a_floor_usd"], rcv["rebuild_cost_usd"], home_value)

    coverages: list[RecommendedCoverage] = [
        RecommendedCoverage(
            type="Coverage A - Dwelling",
            limit=str(int(cov_a)),
            rationale=f"Cov A = max(lender floor, rebuild cost, home value) = ${int(cov_a):,}; settlement {floor['settlement_basis']}.",
            iso_code="COV-A",
            evidence_ids=floor["evidence_ids"] + [rcv["evidence_id"]],
        ),
        RecommendedCoverage(
            type="Coverage B - Other Structures",
            limit=str(int(cov_a * 0.10)),
            rationale="Default 10% of Coverage A per HO-3 form.",
            iso_code="COV-B",
            evidence_ids=["NAIC-CG-COVERAGE-01"],
        ),
        RecommendedCoverage(
            type="Coverage C - Personal Property",
            limit=str(int(cov_a * 0.50)),
            rationale="Default 50% of Coverage A; raise via scheduled property if jewellery/art exposure.",
            iso_code="COV-C",
            evidence_ids=["NAIC-CG-COVERAGE-01"],
        ),
        RecommendedCoverage(
            type="Coverage D - Loss of Use",
            limit="24 months" if state_code == "CA" else "12 months",
            rationale=("Cal. Ins. Code §2051.5 minimum 24 months following declared disaster" if state_code == "CA" else "Standard HO-3 12-month default"),
            iso_code="COV-D",
            evidence_ids=["RULE-CA-COVD-MIN-24MO"] if state_code == "CA" else ["NAIC-CG-COVERAGE-01"],
        ),
        RecommendedCoverage(
            type="Coverage E - Personal Liability",
            limit="300000",
            rationale="Recommended liability limit; uplifted from $100K default given pool/dog exposures.",
            iso_code="COV-E",
            evidence_ids=["III-HB-PRICING-01"],
        ),
    ]

    # ---- Step 2: peril coverages ---------------------------------------------
    if state_code == "CA":
        cea = cea_earthquake_recommender.invoke({
            "coverage_a_usd": cov_a,
            "year_built": sanitized.get("year_built"),
            "foundation": sanitized.get("foundation"),
        })
        coverages.append(
            RecommendedCoverage(
                type="Endorsement - CEA Earthquake Companion",
                limit=f"{cea['recommended_deductible_pct']}% deductible (${int(cea['recommended_deductible_usd']):,})",
                rationale=cea["rationale"],
                iso_code="END-EQ-CEA",
                evidence_ids=[cea["evidence_id"], "RULE-CA-EQ-OFFER"],
            )
        )

    if state_code == "FL":
        coverages.append(
            RecommendedCoverage(
                type="Endorsement - Catastrophic Ground Cover Collapse",
                limit="full Coverage A",
                rationale="Mandatory CGCC inclusion per Fla. Stat. §627.706.",
                iso_code="END-CGCC",
                evidence_ids=["RULE-FL-CGCC-MANDATORY"],
            )
        )
        hurr = fl_hurricane_deductible.invoke({"coverage_a_usd": cov_a})
        for opt in hurr["options"]:
            if opt["is_eligible"]:
                coverages.append(
                    RecommendedCoverage(
                        type=f"Hurricane Deductible Option ({opt['deductible_label']})",
                        limit=opt["deductible_label"],
                        rationale=f"Statutorily-mandated option per Fla. Stat. §627.701; premium factor {opt['premium_factor']}×.",
                        iso_code="END-HURR-DED",
                        evidence_ids=[opt["evidence_id"], "RULE-FL-HURRICANE-DEDUCTIBLE"],
                    )
                )
        wind = wind_mitigation_discount.invoke({})  # default - 0% since unknown construction
        coverages.append(
            RecommendedCoverage(
                type="Wind Mitigation Inspection (advisory)",
                limit="up to 45% wind premium discount",
                rationale="Recommend OIR-B1-1802 inspection; documented mitigation features unlock the wind discount.",
                iso_code="WIND-FORM",
                evidence_ids=[wind["evidence_id"], "RULE-FL-WIND-MITIGATION"],
            )
        )

    # NFIP if mandated
    if rules["required_coverages"] and "flood_nfip_or_private_equivalent" in rules["required_coverages"]:
        coverages.append(
            RecommendedCoverage(
                type="Flood (NFIP or private equivalent)",
                limit="dwelling RCV up to NFIP cap (250K)",
                rationale="Mandatory due to SFHA + mortgage (FDPA 1973).",
                iso_code="COV-FLOOD-NFIP",
                evidence_ids=["RULE-NFIP-MANDATORY"],
            )
        )

    # ---- Step 3+4: Optional LLM polish pass for rationale prose --------------
    try:
        llm = get_llm(AgentRole.COVERAGE_PLANNER)
        plan_obj = llm.with_structured_output(CoveragePlan).invoke(
            [
                {"role": "system", "content": prompts.COVERAGE_PLANNER},
                {
                    "role": "user",
                    "content": (
                        f"State {state_code}, home_value ${int(home_value):,}, has_mortgage={has_mortgage}, in_sfha={in_sfha}, "
                        f"risk_factors={[rf.factor for rf in state.get('risk_factors', [])]}. "
                        f"required_coverages={rules['required_coverages']}, required_offers={rules['required_offers']}, "
                        f"floors={rules['floors']}. Produce the 4-step plan; the executor will run it deterministically."
                    ),
                },
            ]
        )
    except Exception:
        plan_obj = None

    node = make_node(
        agent="CoverageAgent",
        summary=f"Plan-and-Execute -> {len(coverages)} coverage line(s).",
        evidence_ids=list({eid for c in coverages for eid in c.evidence_ids}),
        payload={
            "plan": [s.model_dump() for s in (plan_obj.steps if plan_obj else [])],
            "coverage_a_usd": int(cov_a),
        },
    )
    return {
        "recommended_coverages": coverages,
        "decision_trace": append_node(list(state.get("decision_trace", [])), node),
    }
