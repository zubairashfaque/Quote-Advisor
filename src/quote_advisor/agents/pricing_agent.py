"""Pricing Agent (ReWOO).

Three nodes:
  1. ``pricing_planner_node``  - one LLM call, emits a DAG of multiplier-lookup tasks
  2. ``pricing_worker_node``   - per-task deterministic tool dispatch (Send-fanned in graph.py)
  3. ``pricing_solver_node``   - one LLM call, composes the chain and emits PremiumRange

Workers carry no LLM call - the heavy lifting is one planner LLM + one solver LLM
plus the parallel deterministic tool calls.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .. import prompts
from ..decision_trace import append_node, make_node
from ..llm_registry import AgentRole, get_llm
from ..schemas import FactorChainEntry, PremiumRange
from ..tools import (
    base_premium,
    citizens_benchmark,
    cohort_benchmark,
    home_value_scaling_factor,
    pricing_multiplier_lookup,
)


# =============================================================================
# Plan / Worker / Solver schemas
# =============================================================================


class PricingTask(BaseModel):
    step_id: str
    tool: str  # 'base_premium' | 'pricing_multiplier_lookup' | 'home_value_scaling_factor' | ...
    inputs: dict[str, Any]
    depends_on: list[str] = Field(default_factory=list)


class PricingPlan(BaseModel):
    tasks: list[PricingTask]


class PricingTaskResult(BaseModel):
    step_id: str
    tool: str
    output: dict[str, Any]


# =============================================================================
# Planner node
# =============================================================================


def _deterministic_plan(
    state_code: str,
    home_value: float,
    risk_severities: dict[str, str],
    field_treatments: dict[str, str],
    claims_history: int,
    has_pool: bool,
    county_fips: str | None,
) -> list[PricingTask]:
    tasks: list[PricingTask] = [
        PricingTask(step_id="#E1", tool="base_premium", inputs={"state": state_code, "year": 2026}),
        PricingTask(step_id="#E2", tool="home_value_scaling_factor", inputs={"home_value_usd": home_value}),
    ]

    next_id = 3
    for peril, sev in risk_severities.items():
        tasks.append(
            PricingTask(
                step_id=f"#E{next_id}",
                tool="pricing_multiplier_lookup",
                inputs={"dimension": peril, "key": sev},
            )
        )
        next_id += 1

    tasks.append(
        PricingTask(
            step_id=f"#E{next_id}",
            tool="pricing_multiplier_lookup",
            inputs={"dimension": "claims", "key": str(claims_history) if claims_history < 3 else "3+"},
        )
    )
    next_id += 1

    if has_pool:
        tasks.append(
            PricingTask(step_id=f"#E{next_id}", tool="pricing_multiplier_lookup",
                        inputs={"dimension": "pool", "key": "true"})
        )
        next_id += 1

    credit_treatment = field_treatments.get("credit_score") or "neutral_1.0x"
    credit_key = "neutral_1.0x" if credit_treatment.startswith("neutral") else "dropped" if credit_treatment.startswith("dropped") else "neutral_1.0x"
    tasks.append(
        PricingTask(step_id=f"#E{next_id}", tool="pricing_multiplier_lookup",
                    inputs={"dimension": "credit_score", "key": credit_key})
    )
    next_id += 1

    if state_code == "FL" and county_fips:
        tasks.append(
            PricingTask(step_id=f"#E{next_id}", tool="citizens_benchmark",
                        inputs={"state": state_code, "county_fips": county_fips, "coastal_distance_band": "coastal_0_5mi"})
        )
        next_id += 1

    cohort_inputs: dict[str, Any] = {"state": state_code, "home_value_usd": home_value}
    if state_code == "CA":
        cohort_inputs["wildfire_tier"] = _sev_to_tier(risk_severities.get("wildfire", "low"))
    elif state_code == "FL":
        cohort_inputs["hurricane_tier"] = _sev_to_tier(risk_severities.get("hurricane", "low"))
    tasks.append(PricingTask(step_id=f"#E{next_id}", tool="cohort_benchmark", inputs=cohort_inputs))
    return tasks


def _sev_to_tier(sev: str) -> str:
    return {"low": "Low", "medium": "Moderate", "high": "Very High"}.get(sev.lower(), "Moderate")


def pricing_planner_node(state: dict[str, Any]) -> dict[str, Any]:
    """LLM-backed planner with deterministic fallback."""
    sanitized = state.get("sanitized_profile") or {}
    state_code = (sanitized.get("state") or "").upper()
    home_value = float(sanitized.get("home_value") or 0)
    claims_history = int(sanitized.get("claims_history") or 0)
    has_pool = bool(sanitized.get("has_pool"))
    field_treatments = state.get("field_treatments") or {}
    county_fips = sanitized.get("county_fips") or ("12086" if state_code == "FL" else "06037" if state_code == "CA" else None)

    risk_severities: dict[str, str] = {}
    for rf in state.get("risk_factors", []) or []:
        name = rf.factor.lower() if hasattr(rf, "factor") else str(rf.get("factor", "")).lower()
        sev = rf.severity.value if hasattr(rf, "severity") else str(rf.get("severity", "low"))
        if "wildfire" in name:
            risk_severities["wildfire"] = sev
        elif "seismic" in name or "earthquake" in name:
            risk_severities["seismic"] = sev
        elif "hurricane" in name:
            risk_severities["hurricane"] = sev
        elif "flood" in name:
            risk_severities["flood"] = sev

    plan_tasks = _deterministic_plan(state_code, home_value, risk_severities, field_treatments,
                                     claims_history, has_pool, county_fips)

    node = make_node(
        agent="PricingPlanner",
        summary=f"ReWOO plan emitted with {len(plan_tasks)} tasks.",
        payload={"plan": [t.model_dump() for t in plan_tasks]},
    )
    return {
        "pricing_plan": [t.model_dump() for t in plan_tasks],
        "decision_trace": append_node(list(state.get("decision_trace", [])), node),
    }


# =============================================================================
# Worker node - dispatched per task via LangGraph Send (set up in graph.py)
# =============================================================================


_TOOL_DISPATCH = {
    "base_premium":               base_premium,
    "home_value_scaling_factor":  home_value_scaling_factor,
    "pricing_multiplier_lookup":  pricing_multiplier_lookup,
    "citizens_benchmark":         citizens_benchmark,
    "cohort_benchmark":           cohort_benchmark,
}


def pricing_worker_node(task_payload: dict[str, Any]) -> dict[str, Any]:
    """Execute one task. Receives a dict with {step_id, tool, inputs}. Returns {pricing_results: [...]}."""
    step_id = task_payload["step_id"]
    tool_name = task_payload["tool"]
    inputs = task_payload["inputs"]
    fn = _TOOL_DISPATCH[tool_name]
    try:
        out = fn.invoke(inputs)
    except Exception as exc:
        out = {"error": repr(exc)}
    return {"pricing_results": [{"step_id": step_id, "tool": tool_name, "output": out}]}


# =============================================================================
# Solver node
# =============================================================================


class SolverNarrative(BaseModel):
    factor_chain: list[FactorChainEntry]
    point_estimate_usd: float
    range_low_usd: float
    range_high_usd: float
    explanation: str


def pricing_solver_node(state: dict[str, Any]) -> dict[str, Any]:
    """Compose the chain into a PremiumRange + factor_chain.

    Idempotency note: ``pricing_results`` uses an ``operator.add`` reducer in
    GraphState so parallel ReWOO Workers can fan-in via Send within a single
    run. Across re-invocations on the same ``thread_id``, MemorySaver restores
    the prior list, so a naive read would compound multipliers (e.g. 2.17^N
    home-value scaling after N re-runs). We dedupe by ``step_id`` and keep
    only the latest occurrence so re-runs are stable.
    """
    raw_results: list[dict[str, Any]] = state.get("pricing_results", []) or []
    by_step: dict[str, dict[str, Any]] = {}
    for r in raw_results:
        sid = r.get("step_id")
        if sid is None:
            continue  # ignore malformed entries
        by_step[sid] = r  # keep last write per step_id (latest run wins)
    results: list[dict[str, Any]] = list(by_step.values())

    # Locate base + scaling.
    base_val = 0.0
    base_evidence = "BENCH-DEFAULT"
    scaling_val = 1.0
    chain: list[FactorChainEntry] = []
    cohort: dict[str, Any] | None = None
    citizens_row: dict[str, Any] | None = None

    for r in results:
        out = r.get("output", {}) or {}
        tool_name = r.get("tool")
        if tool_name == "base_premium":
            base_val = float(out.get("base_premium_usd", 0))
            base_evidence = out.get("evidence_id", "BENCH-DEFAULT")
            chain.append(FactorChainEntry(
                name=f"Base premium ({out.get('state', '?')} {out.get('year', '?')})",
                multiplier=1.0,
                evidence_id=base_evidence,
                rationale=out.get("source", ""),
            ))
        elif tool_name == "home_value_scaling_factor":
            scaling_val = float(out.get("scaling_multiplier", 1.0))
            chain.append(FactorChainEntry(
                name="Home-value scaling",
                multiplier=scaling_val,
                evidence_id=out.get("evidence_id", "MULT-SCALING-PER100K"),
                rationale=out.get("explanation", ""),
            ))
        elif tool_name == "pricing_multiplier_lookup":
            mult = float(out.get("multiplier", 1.0))
            ev_id = out.get("evidence_id", "MULT-UNK")
            # Skip unbound placeholder slots (planner emitted a task the worker
            # couldn't satisfy: no name, no real multiplier, no real evidence).
            # These contributed `? (?) = 1.00x · MULT-UNK` rows that leaked into
            # the user-facing explanation. Filter them at source.
            if ev_id == "MULT-UNK" and mult == 1.0:
                continue
            chain.append(FactorChainEntry(
                name=f"{out.get('dimension', '?')} ({out.get('key', '?')})",
                multiplier=mult,
                evidence_id=ev_id,
            ))
        elif tool_name == "cohort_benchmark":
            cohort = out
        elif tool_name == "citizens_benchmark":
            citizens_row = out

    # Compose chain.
    point = base_val
    for entry in chain:
        if entry.name.startswith("Base premium"):
            continue
        point *= entry.multiplier
    low = round(point * 0.75, 2)
    high = round(point * 1.25, 2)

    # Optional LLM rationale prose (terse).
    explanation = (
        f"Premium chain: ${base_val:,.0f} base × scaling {scaling_val:.2f} × "
        + " × ".join(f"{e.multiplier:.2f} ({e.name})" for e in chain if e.name != f"Base premium ({state.get('sanitized_profile', {}).get('state', '?')} 2026)" and e.name != "Home-value scaling")
        + f" = ${point:,.0f} point estimate; +/-25% gives [${low:,.0f}, ${high:,.0f}]."
    )
    if cohort:
        explanation += f" Cohort p50=${cohort.get('p50_premium_usd', 0):,.0f}, p90=${cohort.get('p90_premium_usd', 0):,.0f} ({cohort.get('evidence_id', '?')})."
    if citizens_row:
        explanation += (
            f" Citizens benchmark ${citizens_row.get('base_actuarial_per_1000_cov_a', 0):.2f}/$1000 of CovA "
            f"({citizens_row.get('evidence_id', '?')})."
        )

    # Optional LLM polish - PROSE ONLY. Deterministic dollar amounts (point/low/high)
    # MUST come from the chain math above; the LLM cannot override them. We also
    # keep the deterministic chain entries because the LLM has been observed to
    # rename / replace evidence_ids when given the chance.
    try:
        llm = get_llm(AgentRole.PRICING_SOLVER).with_structured_output(SolverNarrative)
        polished: SolverNarrative = llm.invoke(
            [
                {"role": "system", "content": prompts.PRICING_SOLVER},
                {
                    "role": "user",
                    "content": (
                        f"Tool outputs: {results}. "
                        f"Composed chain: {[e.model_dump() for e in chain]}. "
                        f"Deterministic point estimate ${point:,.0f}, range [${low:,.0f}, ${high:,.0f}]. "
                        f"Cohort: {cohort}. Citizens: {citizens_row}. "
                        "Return the structured output. The numeric fields (point_estimate_usd / range_low_usd / range_high_usd) "
                        "and the chain entries are authoritative as given - DO NOT alter them; only improve the rationale prose."
                    ),
                },
            ]
        )
        # Prose-only adoption: take the polished explanation if non-empty.
        # Numbers and chain entries are NEVER overridden by the LLM.
        if polished.explanation:
            explanation = polished.explanation
    except Exception:
        pass

    premium = PremiumRange(low=low, high=high, currency="USD")

    node = make_node(
        agent="PricingAgent",
        summary=f"Premium range ${low:,.0f}-${high:,.0f}; chain length {len(chain)}.",
        evidence_ids=[e.evidence_id for e in chain],
        payload={"point_estimate_usd": round(point, 2), "multiplier": scaling_val},
    )
    return {
        "premium_range": premium,
        "pricing_factor_chain": chain,
        "decision_trace": append_node(list(state.get("decision_trace", [])), node),
    }
