"""4-Persona Council protocol (Critic-Refine).

Round 1: each persona produces an independent position in parallel (via
LangGraph Send dispatch in graph.py).
Round 2: each persona sees the others' Round-1 positions and may update.
Final: weighted vote, Compliance VETO is non-overridable.

The Council is convened by the Validator only when deterministic checks fail
or when confidence is below 0.65 (V5 §3.6, §8).
"""

from __future__ import annotations

from typing import Literal

from .schemas import CouncilVerdict, PersonaName, PersonaPosition, PremiumRange


PERSONA_WEIGHTS: dict[PersonaName, float] = {
    PersonaName.UNDERWRITER: 1.0,
    PersonaName.ADVOCATE:    1.0,
    PersonaName.ACTUARY:     1.5,
    PersonaName.COMPLIANCE:  1.5,
}


def aggregate_verdict(
    positions: list[PersonaPosition],
    base_premium_range: PremiumRange | None,
) -> CouncilVerdict:
    """Compute the weighted-vote verdict from Round-2 positions.

    Compliance VETO is non-overridable. When veto fires, premium_range is
    held at base, confidence is later capped at 0.5 in confidence.py, and
    ``refer_to_human`` is set.
    """
    veto = any(p.veto for p in positions if p.persona == PersonaName.COMPLIANCE)
    warnings: list[str] = []
    consensus_kind: Literal["consensus", "majority", "split", "compliance_veto", "all_disagree"]

    if veto:
        compliance = next(p for p in positions if p.persona == PersonaName.COMPLIANCE)
        warnings.append(f"Compliance Officer veto: {compliance.position_summary}")
        return CouncilVerdict(
            invoked=True,
            consensus_kind="compliance_veto",
            final_premium_range=base_premium_range,
            warnings=warnings,
            refer_to_human=True,
            positions=positions,
        )

    proposed = [(p, p.proposed_premium_range) for p in positions if p.proposed_premium_range is not None]
    if not proposed:
        return CouncilVerdict(
            invoked=True,
            consensus_kind="consensus",
            final_premium_range=base_premium_range,
            warnings=warnings,
            refer_to_human=False,
            positions=positions,
        )

    total_w = sum(PERSONA_WEIGHTS[p.persona] * p.weight for p, _ in proposed)
    low = sum((PERSONA_WEIGHTS[p.persona] * p.weight) * pr.low for p, pr in proposed) / total_w
    high = sum((PERSONA_WEIGHTS[p.persona] * p.weight) * pr.high for p, pr in proposed) / total_w
    aggregated = PremiumRange(low=round(low, 2), high=round(high, 2), currency="USD")

    n_proposers = len(proposed)
    if n_proposers == len(positions) and _within_band(proposed, 0.20):
        consensus_kind = "consensus"
    elif n_proposers >= 3:
        consensus_kind = "majority"
        warnings.append("Council majority verdict (3 of 4 proposers); confidence -0.05.")
    elif n_proposers == 2:
        consensus_kind = "split"
        warnings.append("Council split; weighted vote; confidence -0.10.")
    else:
        consensus_kind = "all_disagree"
        warnings.append("Council all-disagree; weighted average used; confidence -0.15.")

    return CouncilVerdict(
        invoked=True,
        consensus_kind=consensus_kind,
        final_premium_range=aggregated,
        warnings=warnings,
        refer_to_human=False,
        positions=positions,
    )


def _within_band(proposed: list[tuple[PersonaPosition, PremiumRange]], band: float) -> bool:
    if not proposed:
        return True
    midpoints = sorted((pr.low + pr.high) / 2 for _, pr in proposed)
    median = midpoints[len(midpoints) // 2]
    if median == 0:
        return True
    return all(abs(m - median) / median <= band for m in midpoints)
