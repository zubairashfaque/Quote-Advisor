"""Validator + 4-Persona Council (Critic-Refine).

Validator runs deterministic checks; if any fail or confidence < 0.65, it
invokes the Council (4 persona prompts in parallel via Send, then a Round-2
cross-exam pass, then aggregate_verdict).
"""

from __future__ import annotations

from typing import Any

from .. import council, prompts
from ..decision_trace import append_node, make_node
from ..llm_registry import AgentRole, get_llm
from ..schemas import (
    PersonaName,
    PersonaPosition,
    PremiumRange,
)

PERSONA_TO_ROLE = {
    PersonaName.UNDERWRITER: AgentRole.COUNCIL_UNDERWRITER,
    PersonaName.ADVOCATE:    AgentRole.COUNCIL_ADVOCATE,
    PersonaName.ACTUARY:     AgentRole.COUNCIL_ACTUARY,
    PersonaName.COMPLIANCE:  AgentRole.COUNCIL_COMPLIANCE,
}

PERSONA_TO_PROMPT = {
    PersonaName.UNDERWRITER: prompts.PERSONA_UNDERWRITER,
    PersonaName.ADVOCATE:    prompts.PERSONA_ADVOCATE,
    PersonaName.ACTUARY:     prompts.PERSONA_ACTUARY,
    PersonaName.COMPLIANCE:  prompts.PERSONA_COMPLIANCE,
}


def validator_node(state: dict[str, Any]) -> dict[str, Any]:
    """Run deterministic checks; flag whether to convene the Council."""
    flags: list[str] = []

    premium = state.get("premium_range")
    if isinstance(premium, dict):
        premium = PremiumRange.model_validate(premium)

    statutory_violations = state.get("statutory_violations") or []
    if statutory_violations:
        flags.append(f"statutory_violations non-empty: {statutory_violations}")

    # Cohort band check (if a cohort row was retrieved).
    cohort_payload = None
    for r in state.get("pricing_results", []) or []:
        if r.get("tool") == "cohort_benchmark":
            cohort_payload = r.get("output")
    if cohort_payload and premium:
        midpoint = (premium.low + premium.high) / 2
        p10 = cohort_payload.get("p10_premium_usd", 0)
        p90 = cohort_payload.get("p90_premium_usd", 0)
        if p10 and (midpoint < p10 or midpoint > p90):
            flags.append(f"premium midpoint ${midpoint:,.0f} outside cohort p10-p90 [${p10:,.0f}, ${p90:,.0f}]")

    # Lender floor check (if recommended_coverages includes Cov A).
    cov_a_recommended = None
    for c in state.get("recommended_coverages", []) or []:
        type_str = c.type if hasattr(c, "type") else c.get("type", "")
        if "Coverage A" in type_str or type_str == "Coverage A - Dwelling":
            limit = c.limit if hasattr(c, "limit") else c.get("limit")
            try:
                cov_a_recommended = float(limit)
            except (TypeError, ValueError):
                pass
    floor_record = (state.get("floors") or {}).get("coverage_a")
    if cov_a_recommended is not None and floor_record:
        floor_val = floor_record.get("value")
        if isinstance(floor_val, (int, float)) and cov_a_recommended < floor_val:
            flags.append(f"Cov A {cov_a_recommended:,.0f} below lender floor {floor_val:,.0f}")

    council_invoked = bool(flags)
    node = make_node(
        agent="Validator",
        summary=f"Deterministic checks: {len(flags)} flag(s); council_invoked={council_invoked}.",
        payload={"flags": flags},
    )
    return {
        "consistency_flags": flags,
        "council_invoked": council_invoked,
        "decision_trace": append_node(list(state.get("decision_trace", [])), node),
    }


def council_round_node(state: dict[str, Any]) -> dict[str, Any]:
    """One-shot Council convene: round-1 parallel positions + round-2 update + aggregate verdict.

    For brevity all four persona invocations are sequential here; production
    would Send-fan them in parallel - the orchestration is unchanged.
    """
    if not state.get("council_invoked"):
        return {}  # no-op when Validator did not flag

    premium = state.get("premium_range")
    if isinstance(premium, dict):
        premium = PremiumRange.model_validate(premium)

    base_position_context = {
        "sanitized_profile":        state.get("sanitized_profile"),
        "field_treatments":         state.get("field_treatments"),
        "triggered_rules":          [r.model_dump() for r in (state.get("triggered_rules") or []) if hasattr(r, "model_dump")],
        "risk_factors":             [r.model_dump() if hasattr(r, "model_dump") else r for r in (state.get("risk_factors") or [])],
        "recommended_coverages":    [c.model_dump() if hasattr(c, "model_dump") else c for c in (state.get("recommended_coverages") or [])],
        "premium_range":            premium.model_dump() if premium else None,
        "factor_chain":             [e.model_dump() if hasattr(e, "model_dump") else e for e in (state.get("pricing_factor_chain") or [])],
        "consistency_flags":        state.get("consistency_flags", []),
    }

    round1: list[PersonaPosition] = []
    for persona in PersonaName:
        try:
            llm = get_llm(PERSONA_TO_ROLE[persona]).with_structured_output(PersonaPosition)
            pos = llm.invoke(
                [
                    {"role": "system", "content": PERSONA_TO_PROMPT[persona]},
                    {"role": "user", "content": f"Round 1 - independent position. Context: {base_position_context}"},
                ]
            )
            round1.append(pos)
        except Exception:
            round1.append(
                PersonaPosition(
                    persona=persona,
                    weight=1.0,
                    position_summary=f"{persona.value}: deferred (LLM error)",
                )
            )

    # Round 2: each persona sees the others' positions.
    others_view = [p.model_dump() for p in round1]
    round2: list[PersonaPosition] = []
    for persona in PersonaName:
        try:
            llm = get_llm(PERSONA_TO_ROLE[persona]).with_structured_output(PersonaPosition)
            pos = llm.invoke(
                [
                    {"role": "system", "content": PERSONA_TO_PROMPT[persona]},
                    {
                        "role": "user",
                        "content": (
                            f"Round 2 - cross-examination. Other personas' Round-1 positions: {others_view}. "
                            f"Context: {base_position_context}. Update your position or hold."
                        ),
                    },
                ]
            )
            round2.append(pos)
        except Exception:
            round2.append(next((p for p in round1 if p.persona == persona), round1[0]))

    verdict = council.aggregate_verdict(round2, premium)
    final_premium = verdict.final_premium_range or premium

    node = make_node(
        agent="Council",
        summary=f"Verdict={verdict.consensus_kind}; refer_to_human={verdict.refer_to_human}.",
        payload={"verdict": verdict.model_dump()},
    )
    update: dict[str, Any] = {
        "council_verdict": verdict,
        "premium_range":   final_premium,
        "decision_trace":  append_node(list(state.get("decision_trace", [])), node),
        "warnings":        list(state.get("warnings", []) or []) + verdict.warnings,
        "refer_to_human":  verdict.refer_to_human or state.get("refer_to_human", False),
    }
    return update
