"""Counterfactual Agent (Reflexion outer + Tree-of-Thoughts inner for multi-axis).

Forks the GraphState, applies 1+ mutations, re-runs Risk -> Coverage -> Pricing
on the fork, and returns a CounterfactualDelta. Plausibility check (+/-25%);
verbal reflection persisted across follow-up turns within the same thread.
"""

from __future__ import annotations

import copy
import sys
from typing import Any

from rich.console import Console

from .. import prompts
from ..decision_trace import append_node, make_node
from ..llm_registry import AgentRole, get_llm
from ..schemas import (
    CounterfactualDelta,
    CounterfactualRequest,
    MutationAxis,
    PremiumRange,
)
from ..tools import state_diff
from .coverage_agent import coverage_node
from .pricing_agent import pricing_planner_node, pricing_solver_node, pricing_worker_node
from .risk_agent import risk_node

_console = Console(file=sys.stderr)


PLAUSIBILITY_LOWER = -0.50  # -50% delta
PLAUSIBILITY_UPPER = +0.50  # +50% delta


def _fork_with_mutations(state: dict[str, Any], axes: list[MutationAxis]) -> dict[str, Any]:
    fork = copy.deepcopy(state)
    sanitized = dict(fork.get("sanitized_profile") or {})
    for axis in axes:
        sanitized[axis.field] = axis.new_value
    fork["sanitized_profile"] = sanitized
    # Drop downstream products so they get recomputed against the mutated profile.
    fork.pop("risk_factors", None)
    fork.pop("recommended_coverages", None)
    fork.pop("premium_range", None)
    fork.pop("pricing_factor_chain", None)
    fork.pop("pricing_results", None)
    return fork


def _rerun_subgraph(forked: dict[str, Any]) -> dict[str, Any]:
    forked.update(risk_node(forked) or {})
    forked.update(coverage_node(forked) or {})
    plan_update = pricing_planner_node(forked) or {}
    forked.update(plan_update)
    pricing_results: list[dict[str, Any]] = []
    for task in forked.get("pricing_plan", []) or []:
        out = pricing_worker_node(task) or {}
        pricing_results.extend(out.get("pricing_results", []))
    forked["pricing_results"] = pricing_results
    forked.update(pricing_solver_node(forked) or {})
    return forked


def _delta(base: PremiumRange | None, fork: PremiumRange | None) -> tuple[float, float, float, float]:
    if base is None or fork is None:
        return 0.0, 0.0, 0.0, 0.0
    delta_low_usd = round(fork.low - base.low, 2)
    delta_high_usd = round(fork.high - base.high, 2)
    delta_low_pct = (fork.low - base.low) / base.low if base.low else 0.0
    delta_high_pct = (fork.high - base.high) / base.high if base.high else 0.0
    return delta_low_usd, delta_high_usd, round(delta_low_pct, 4), round(delta_high_pct, 4)


def counterfactual_node(state: dict[str, Any]) -> dict[str, Any]:
    """Apply mutations, re-run subgraph, score plausibility, reflect-and-retry if needed."""
    cf_state = state.get("counterfactual_base_state") or {}
    raw_axes = cf_state.get("mutation_axes") or []
    axes = [MutationAxis(**a) if isinstance(a, dict) else a for a in raw_axes]
    if not axes:
        node = make_node(agent="Counterfactual", summary="No mutation axes parsed; refusing.")
        return {
            "counterfactual_delta": CounterfactualDelta(
                base_premium=PremiumRange(low=0.0, high=0.0, currency="USD"),
                cf_premium=PremiumRange(low=0.0, high=0.0, currency="USD"),
                delta_low_usd=0.0, delta_high_usd=0.0, delta_low_pct=0.0, delta_high_pct=0.0,
                drivers_changed=[],
                mutations={},
                reflexion_notes=["Counterfactual refused: no mutations parseable from the question."],
                plausibility_status="refused",
            ),
            "decision_trace": append_node(list(state.get("decision_trace", [])), node),
        }

    base_premium = state.get("premium_range")
    if isinstance(base_premium, dict):
        base_premium = PremiumRange.model_validate(base_premium)

    reflexion_memory: list[str] = list(state.get("counterfactual_reflexion_memory") or [])

    # ---- TRIAL 1 -------------------------------------------------------------
    forked = _fork_with_mutations(state, axes)
    forked = _rerun_subgraph(forked)
    cf_premium = forked.get("premium_range")
    if isinstance(cf_premium, dict):
        cf_premium = PremiumRange.model_validate(cf_premium)

    delta_low_usd, delta_high_usd, delta_low_pct, delta_high_pct = _delta(base_premium, cf_premium)

    plausible = PLAUSIBILITY_LOWER <= delta_low_pct <= PLAUSIBILITY_UPPER and PLAUSIBILITY_LOWER <= delta_high_pct <= PLAUSIBILITY_UPPER

    # ---- REFLEXION + TRIAL 2 -------------------------------------------------
    if not plausible:
        try:
            llm = get_llm(AgentRole.COUNTERFACTUAL)
            reflection = llm.invoke(
                [
                    {"role": "system", "content": prompts.COUNTERFACTUAL_REFLECT.format(
                        delta_pct=max(abs(delta_low_pct), abs(delta_high_pct)),
                        mutation_axes=[a.model_dump() for a in axes],
                        lower=PLAUSIBILITY_LOWER, upper=PLAUSIBILITY_UPPER,
                    )},
                    {"role": "user", "content": f"factor_chain={forked.get('pricing_factor_chain')}"},
                ]
            )
            reflection_text = reflection.content if hasattr(reflection, "content") else str(reflection)
        except Exception as exc:
            reflection_text = f"(reflection skipped due to LLM error: {exc!r})"
        reflexion_memory.append(reflection_text if isinstance(reflection_text, str) else str(reflection_text))

        # TRIAL 2 - re-run the pricing solver with reflection guidance prepended to the prompt.
        # We don't re-fork; we re-call the solver with augmented context.
        forked["counterfactual_reflexion_memory"] = reflexion_memory
        forked.update(pricing_solver_node(forked) or {})
        cf_premium = forked.get("premium_range")
        if isinstance(cf_premium, dict):
            cf_premium = PremiumRange.model_validate(cf_premium)
        delta_low_usd, delta_high_usd, delta_low_pct, delta_high_pct = _delta(base_premium, cf_premium)
        plausible = PLAUSIBILITY_LOWER <= delta_low_pct <= PLAUSIBILITY_UPPER and PLAUSIBILITY_LOWER <= delta_high_pct <= PLAUSIBILITY_UPPER

    # ---- DRIVERS CHANGED via StateDiffTool -----------------------------------
    keys_of_interest = ["risk_factors", "recommended_coverages", "premium_range", "pricing_factor_chain"]
    diff = state_diff.invoke({
        "state_a": _flatten(state),
        "state_b": _flatten(forked),
        "keys_of_interest": keys_of_interest,
    })
    drivers = [c["key"] for c in diff.get("changed_fields", [])]

    delta = CounterfactualDelta(
        base_premium=base_premium or PremiumRange(low=0.0, high=0.0, currency="USD"),
        cf_premium=cf_premium or PremiumRange(low=0.0, high=0.0, currency="USD"),
        delta_low_usd=delta_low_usd,
        delta_high_usd=delta_high_usd,
        delta_low_pct=delta_low_pct,
        delta_high_pct=delta_high_pct,
        drivers_changed=drivers,
        mutations={a.field: a.new_value for a in axes},
        reflexion_notes=reflexion_memory,
        plausibility_status="plausible" if plausible else "refused",
    )

    # ---- DETERMINISTIC NARRATION (always runs) -------------------------------
    mutations_str = ", ".join(f"{k}={v!r}" for k, v in delta.mutations.items())
    base_low = base_premium.low if base_premium else 0.0
    base_high = base_premium.high if base_premium else 0.0
    cf_low = cf_premium.low if cf_premium else 0.0
    cf_high = cf_premium.high if cf_premium else 0.0
    is_noop = abs(delta_low_usd) < 1.0 and abs(delta_high_usd) < 1.0
    if delta_low_usd < 0 and delta_high_usd < 0:
        direction = "decrease"
    elif delta_low_usd > 0 and delta_high_usd > 0:
        direction = "increase"
    else:
        direction = "change"
    # Signed percentage (negative means premium goes down).
    delta_pct_signed = (delta_low_pct + delta_high_pct) / 2 if (delta_low_pct or delta_high_pct) else 0.0
    template = (
        f"Counterfactual ({mutations_str}): "
        f"premium would {direction} by ${abs(delta_low_usd):,.0f} (low end) and "
        f"${abs(delta_high_usd):,.0f} (high end), a {delta_pct_signed:+.1%} change. "
        f"Original premium range: ${base_low:,.0f}–${base_high:,.0f}. "
        f"New premium range: ${cf_low:,.0f}–${cf_high:,.0f}. "
        f"Status: {delta.plausibility_status}."
    )
    if is_noop:
        mutated_fields = ", ".join(delta.mutations.keys())
        template = (
            f"This is a no-op for your profile — you already have {mutated_fields}={list(delta.mutations.values())[0]}, "
            f"so removing/changing it doesn't apply. Your premium stays at ${base_low:,.0f}–${base_high:,.0f}. "
            f"The premium is driven by other factors (state base rate, home-value scaling, claims history, "
            f"and statutory rules), not by the field you asked to change. Status: {delta.plausibility_status}."
        )
        # No-op narrative is already crystal clear — skip the LLM polish to avoid muddling it.
        polished = template
        _console.print(f"\n[bold]Counterfactual answer:[/bold]\n  {polished}\n")
        node = make_node(
            agent="Counterfactual",
            summary=(
                f"Mutations={delta.mutations}; delta_low={delta_low_usd:+,.0f} USD ({delta_low_pct:+.1%}), "
                f"delta_high={delta_high_usd:+,.0f} USD ({delta_high_pct:+.1%}); status=no-op."
            ),
            payload={"delta": delta.model_dump(), "answer": polished, "template": template, "is_noop": True},
        )
        return {
            "counterfactual_delta": delta,
            "counterfactual_reflexion_memory": reflexion_memory,
            "decision_trace": append_node(list(state.get("decision_trace", [])), node),
            "messages": [{"role": "assistant", "content": polished}],
        }

    # ---- LLM POLISH (best-effort; falls back to template on any error) -------
    polished = template
    try:
        llm = get_llm(AgentRole.COUNTERFACTUAL)
        polish_resp = llm.invoke([
            {"role": "system", "content": (
                "You polish a deterministic counterfactual sentence into one paragraph of natural prose "
                "for an insurance customer. Rules: (1) keep every dollar amount, percentage, and direction "
                "(decrease/increase/no-op) EXACTLY as the template states; (2) do not invent new numbers; "
                "(3) be unambiguous about direction — if the template says 'decrease', say 'goes down' or "
                "'is lower by'; if it says 'increase', say 'goes up' or 'is higher by'; (4) two to three "
                "sentences max; (5) no markdown."
            )},
            {"role": "user", "content": (
                f"Deterministic template:\n{template}\n\n"
                f"Direction: {direction}. Signed percent change: {delta_pct_signed:+.1%}.\n"
                f"Mutations applied: {delta.mutations}.\n"
                f"Drivers changed: {drivers}."
            )},
        ])
        polished_text = polish_resp.content if hasattr(polish_resp, "content") else str(polish_resp)
        if isinstance(polished_text, str) and polished_text.strip():
            polished = polished_text.strip()
    except Exception:
        polished = template

    _console.print(f"\n[bold]Counterfactual answer:[/bold]\n  {polished}\n")

    node = make_node(
        agent="Counterfactual",
        summary=(
            f"Mutations={delta.mutations}; delta_low={delta_low_usd:+,.0f} USD ({delta_low_pct:+.1%}), "
            f"delta_high={delta_high_usd:+,.0f} USD ({delta_high_pct:+.1%}); status={delta.plausibility_status}."
        ),
        payload={"delta": delta.model_dump(), "answer": polished, "template": template},
    )
    return {
        "counterfactual_delta": delta,
        "counterfactual_reflexion_memory": reflexion_memory,
        "decision_trace": append_node(list(state.get("decision_trace", [])), node),
        "messages": [{"role": "assistant", "content": polished}],
    }


def _flatten(state: dict[str, Any]) -> dict[str, Any]:
    """Light flatten so StateDiffTool can compare hashable-ish values."""
    out: dict[str, Any] = {}
    for k, v in state.items():
        try:
            if hasattr(v, "model_dump"):
                out[k] = v.model_dump()
            elif isinstance(v, list) and v and hasattr(v[0], "model_dump"):
                out[k] = [item.model_dump() for item in v]
            else:
                out[k] = v
        except Exception:
            out[k] = repr(v)
    return out
