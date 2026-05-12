"""Statutory Agent (ReAct) — replaces the deterministic StatutoryRulesEngine.

Per DEC-0011 (supersedes DEC-0005). Pipeline:

  Phase 1 · Pre-filter (pure Python)
            - Validate the raw profile via CustomerProfile
            - Normalise state ('California' → 'CA')
            - Short-circuit STATE-SUPPORTED for non-CA/FL profiles

  Phase 2 · ReAct loop
            - create_react_agent with rag_retrieve as the only tool
            - 8-iteration cap, temperature=0
            - Agent decides which corpora to query based on profile signals

  Phase 3 · Structured emission
            - with_structured_output coerces narrative → _StatutoryAgentEmission
            - Concrete typed schema (no dict[str, Any]) for OpenAI strict mode

  Phase 4 · Self-check
            - Every emitted evidence_id must match a chunk that was actually retrieved
            - Rules failing the check are dropped
            - If >50% drop, the entire output is discarded → Phase 5 fires

  Phase 5 · Safety net
            - On any failure (LLM error, RAG outage, malformed output, low grounding),
              fall back to statutory_rules_engine.apply()
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field, ValidationError

from .. import prompts, statutory_rules_engine
from ..decision_trace import append_node, make_node
from ..llm_registry import AgentRole, get_llm
from ..rag.retriever import rag_retrieve
from ..schemas import (
    CustomerProfile,
    RuleFire,
    RuleSeverity,
    StatutoryEngineOutput,
    normalise_state,
)


def _statutory_recursion_limit() -> int:
    """Resolve the LangGraph recursion_limit for the StatutoryAgent from its budget.

    LangGraph counts ~3 supersteps per ReAct iteration (think + tool + observation),
    so we multiply by 3 to match the historic ``24 ~ 8 iterations`` mapping.
    """
    from ..budget_registry import get_budget
    from ..llm_registry import AgentRole
    return get_budget(AgentRole.STATUTORY_AGENT).max_react_iterations * 3


# =============================================================================
# Concrete-typed emission schema (OpenAI strict-mode-friendly)
# =============================================================================


class _EmittedAction(BaseModel):
    """One of 7 action types, all fields optional but type-discriminated."""

    type: Literal[
        "drop_field",
        "flag_field_treatment",
        "require_offer",
        "require_coverage",
        "require_form",
        "set_floor",
        "route_market",
    ]
    field: str | None = None
    treatment: str | None = None
    offer: str | None = None
    coverage: str | None = None
    form: str | None = None
    value: float | None = None
    value_rule: str | None = None
    options: list[str] | None = None
    route_hint: str | None = None
    with_rejection_notice: bool | None = None


class _EmittedRule(BaseModel):
    rule_id: str
    jurisdiction: str
    citation: str
    evidence_id: str
    severity: Literal["mandatory", "advisory"] = "advisory"
    rationale: str
    action: _EmittedAction


class _StatutoryAgentEmission(BaseModel):
    """LLM's structured output — concrete-typed for strict-mode compliance."""

    triggered_rules: list[_EmittedRule] = Field(default_factory=list)
    rationale_summary: str = ""


# =============================================================================
# Public node
# =============================================================================


def statutory_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: replace statutory_gate_node body."""
    raw = state.get("raw_profile") or {}
    if not raw:
        return {}

    # ── Phase 1 · Pre-filter ─────────────────────────────────────────────
    try:
        profile = CustomerProfile.model_validate(raw).with_derived_state()
    except ValidationError as exc:
        return _safety_net(raw, state, reason=f"profile validation failed: {exc!r}")

    state_code = (profile.state or "").upper()
    if state_code not in {"CA", "FL"}:
        # Deterministic short-circuit per system prompt
        return _short_circuit_unsupported(raw, profile, state)

    retrieved_chunks: dict[str, dict[str, Any]] = {}
    react_log: list[dict[str, Any]] = []

    # ── Phase 2 · ReAct loop ─────────────────────────────────────────────
    try:
        narrative = _run_react(profile, retrieved_chunks, react_log)
    except Exception as exc:  # noqa: BLE001
        return _safety_net(raw, state, reason=f"ReAct loop failed: {exc!r}")

    # ── Phase 3 · Structured emission ────────────────────────────────────
    try:
        emission = _coerce(narrative)
    except Exception as exc:  # noqa: BLE001
        return _safety_net(raw, state, reason=f"structured emission failed: {exc!r}")

    if not emission.triggered_rules:
        return _safety_net(raw, state, reason="agent emitted zero rules")

    # ── Phase 4 · Self-check ─────────────────────────────────────────────
    grounded, dropped = _self_check(emission.triggered_rules, retrieved_chunks)
    if dropped and (len(dropped) / max(len(emission.triggered_rules), 1)) > 0.5:
        return _safety_net(
            raw,
            state,
            reason=f"self-check dropped >50% of rules ({len(dropped)}/{len(emission.triggered_rules)})",
        )

    # ── Build StatutoryEngineOutput from grounded rules ──────────────────
    out = _emission_to_output(raw_profile=profile.model_dump(exclude_none=False), rules=grounded)

    audit = make_node(
        agent="StatutoryAgent",
        summary=(
            f"ReAct agent fired {len(grounded)} rule(s) "
            f"({len(react_log)} retrieval(s); {len(dropped)} dropped by self-check)."
        ),
        evidence_ids=[r.evidence_id for r in grounded],
        payload={
            "rule_ids":          [r.rule_id for r in grounded],
            "react_trajectory":  react_log,
            "dropped_rule_ids":  [r.rule_id for r in dropped],
            "rationale_summary": emission.rationale_summary,
            "phase":             "agent",
        },
    )

    return {
        "sanitized_profile":     out.sanitized_profile,
        "triggered_rules":       out.triggered_rules,
        "statutory_violations":  out.statutory_violations,
        "field_treatments":      out.field_treatments,
        "required_offers":       out.required_offers,
        "required_coverages":    out.required_coverages,
        "required_forms":        out.required_forms,
        "floors":                out.floors,
        "market_route_hints":    out.market_route_hints,
        "decision_trace":        append_node(list(state.get("decision_trace", [])), audit),
    }


# =============================================================================
# Phase 2 internals — ReAct loop with rag_retrieve as tool
# =============================================================================


def _run_react(
    profile: CustomerProfile,
    retrieved_chunks: dict[str, dict[str, Any]],
    react_log: list[dict[str, Any]],
) -> str:
    """Run the LangGraph ReAct loop and return the final narrative message."""
    llm = get_llm(AgentRole.STATUTORY_AGENT)

    # We wrap rag_retrieve so we can tap retrievals into retrieved_chunks for self-check.
    tapped_tool = _wrap_rag_retrieve(retrieved_chunks, react_log)

    react_agent = create_react_agent(
        llm,
        tools=[tapped_tool],
        prompt=prompts.STATUTORY_AGENT,
    )

    sanitized_brief = profile.model_dump(exclude_none=False)
    user_msg = (
        f"Customer profile: {sanitized_brief}.\n\n"
        "Retrieve the relevant statute chunks from the corpora your prompt enumerates, "
        "then return a NARRATIVE summary of every triggered rule you identified, including "
        "rule_id, citation, evidence_id (from your retrievals), severity, rationale, and the "
        "action.type plus its parameters. Do not produce JSON yet — that is the next step."
    )

    agent_out = react_agent.invoke(
        {"messages": [{"role": "user", "content": user_msg}]},
        config={"recursion_limit": _statutory_recursion_limit()},
    )
    last = agent_out["messages"][-1].content if agent_out.get("messages") else ""
    return last if isinstance(last, str) else str(last)


def _wrap_rag_retrieve(retrieved_chunks: dict[str, dict[str, Any]], react_log: list[dict[str, Any]]):
    """Return a langchain tool that delegates to rag_retrieve and records what was retrieved."""
    from langchain_core.tools import tool
    from pydantic import BaseModel as _BM
    from typing import Literal as _L

    class _Args(_BM):
        query: str
        corpus: _L["ca_doi", "fl_dfs", "gse_lender"]
        jurisdiction: _L["national", "CA", "FL"]
        top_k: int = 3

    @tool("rag_retrieve", args_schema=_Args)
    def tapped(query: str, corpus: str, jurisdiction: str, top_k: int = 3) -> dict:
        """Retrieve top-k chunks from a jurisdiction-tagged corpus."""
        out = rag_retrieve.invoke(
            {"query": query, "corpus": corpus, "jurisdiction": jurisdiction, "top_k": top_k}
        )
        for c in out.get("chunks", []):
            eid = c.get("evidence_id")
            if eid:
                retrieved_chunks[eid] = c
        react_log.append(
            {
                "query":         query,
                "corpus":        corpus,
                "jurisdiction":  jurisdiction,
                "top_k":         top_k,
                "n_chunks":      len(out.get("chunks", [])),
                "evidence_ids":  [c.get("evidence_id") for c in out.get("chunks", [])],
            }
        )
        return out

    return tapped


# =============================================================================
# Phase 3 — coerce narrative → structured emission
# =============================================================================


def _coerce(narrative: str) -> _StatutoryAgentEmission:
    """Second LLM call: convert the ReAct narrative into the structured emission schema."""
    llm = get_llm(AgentRole.STATUTORY_AGENT).with_structured_output(_StatutoryAgentEmission)
    return llm.invoke(
        [
            {
                "role": "system",
                "content": (
                    "Convert the following narrative into a _StatutoryAgentEmission. "
                    "Preserve every evidence_id exactly. Use the action.type strings: "
                    "drop_field, flag_field_treatment, require_offer, require_coverage, "
                    "require_form, set_floor, route_market."
                ),
            },
            {"role": "user", "content": narrative},
        ]
    )


# =============================================================================
# Phase 4 — self-check: every evidence_id must trace to a retrieved chunk
# =============================================================================


def _self_check(
    rules: list[_EmittedRule],
    retrieved_chunks: dict[str, dict[str, Any]],
) -> tuple[list[_EmittedRule], list[_EmittedRule]]:
    """Drop rules whose evidence_id was never returned by rag_retrieve.

    Exception: a small allowlist of evidence_ids that are well-established
    federal statutes whose corpus chunk doesn't exist yet (NFIP, GSE-NFIP overlap).
    """
    _STATUTORY_ALLOWLIST = {
        "RULE-NFIP-MANDATORY",     # 42 U.S.C. §4012a — corpus has FEMA P-312 but not the statute itself
    }
    grounded: list[_EmittedRule] = []
    dropped: list[_EmittedRule] = []
    for r in rules:
        if r.evidence_id in retrieved_chunks or r.evidence_id in _STATUTORY_ALLOWLIST:
            grounded.append(r)
        else:
            dropped.append(r)
    return grounded, dropped


# =============================================================================
# Convert grounded LLM emission → full StatutoryEngineOutput (8 fields)
# =============================================================================


def _emission_to_output(
    raw_profile: dict[str, Any],
    rules: list[_EmittedRule],
) -> StatutoryEngineOutput:
    """Replay each rule's action against the sanitized_profile, building the 8-field output.

    This is the same logic as statutory_rules_engine._apply_action but applied to LLM-produced
    rules. Determinism is preserved — the LLM only proposes the rule list; the action execution
    is pure Python.
    """
    sanitized = dict(raw_profile)
    out = StatutoryEngineOutput(sanitized_profile=sanitized)

    for r in rules:
        a = r.action
        atype = a.type
        try:
            if atype == "drop_field" and a.field:
                sanitized[a.field] = None
                out.field_treatments[a.field] = f"dropped_by_{r.rule_id}"
            elif atype == "flag_field_treatment" and a.field and a.treatment:
                out.field_treatments[a.field] = a.treatment
            elif atype == "require_offer" and a.offer:
                offer_payload: dict[str, Any] = {
                    "rule_id":     r.rule_id,
                    "evidence_id": r.evidence_id,
                    "offer":       a.offer,
                }
                if a.options:
                    offer_payload["options"] = a.options
                if a.form:
                    offer_payload["form"] = a.form
                if a.with_rejection_notice is not None:
                    offer_payload["with_rejection_notice"] = a.with_rejection_notice
                out.required_offers.append(offer_payload)
            elif atype == "require_coverage" and a.coverage:
                out.required_coverages.append(a.coverage)
            elif atype == "require_form" and a.form:
                out.required_forms.append(a.form)
            elif atype == "set_floor" and a.field:
                floor: dict[str, Any] = {"rule_id": r.rule_id, "evidence_id": r.evidence_id}
                if a.value is not None:
                    floor["value"] = a.value
                if a.value_rule:
                    floor["value_rule"] = a.value_rule
                out.floors[a.field] = floor
            elif atype == "route_market" and a.route_hint:
                out.market_route_hints.append(a.route_hint)
            else:
                # Action shape didn't carry the required field — record but don't crash.
                out.statutory_violations.append(
                    f"Agent emitted incomplete action for rule {r.rule_id}: type={atype}"
                )
                continue
        except Exception as exc:  # noqa: BLE001
            out.statutory_violations.append(
                f"Engine failure replaying agent rule {r.rule_id}: {exc!r}"
            )
            continue

        out.triggered_rules.append(
            RuleFire(
                rule_id=r.rule_id,
                jurisdiction=r.jurisdiction,
                citation=r.citation,
                evidence_id=r.evidence_id,
                severity=RuleSeverity(r.severity),
                rationale=r.rationale,
                action=a.model_dump(exclude_none=True),
            )
        )

    out.sanitized_profile = sanitized
    return out


# =============================================================================
# Phase 5 — deterministic safety net
# =============================================================================


def _safety_net(raw: dict[str, Any], state: dict[str, Any], *, reason: str) -> dict[str, Any]:
    """Fall back to the legacy deterministic engine; pipeline never ships malformed output."""
    out = statutory_rules_engine.apply(raw)
    audit = make_node(
        agent="StatutoryAgent",
        summary=f"[FALLBACK] LLM agent did not produce usable output ({reason}); legacy engine fired {len(out.triggered_rules)} rule(s).",
        evidence_ids=[r.evidence_id for r in out.triggered_rules],
        payload={
            "rule_ids": [r.rule_id for r in out.triggered_rules],
            "phase":    "fallback",
            "reason":   reason,
        },
    )
    return {
        "sanitized_profile":     out.sanitized_profile,
        "triggered_rules":       out.triggered_rules,
        "statutory_violations":  out.statutory_violations,
        "field_treatments":      out.field_treatments,
        "required_offers":       out.required_offers,
        "required_coverages":    out.required_coverages,
        "required_forms":        out.required_forms,
        "floors":                out.floors,
        "market_route_hints":    out.market_route_hints,
        "decision_trace":        append_node(list(state.get("decision_trace", [])), audit),
    }


# =============================================================================
# Deterministic STATE-SUPPORTED short-circuit
# =============================================================================


def _short_circuit_unsupported(
    raw: dict[str, Any],
    profile: CustomerProfile,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Skip the LLM entirely for out-of-jurisdiction profiles."""
    sanitized = profile.model_dump(exclude_none=False)
    rule = RuleFire(
        rule_id="STATE-SUPPORTED",
        jurisdiction="*",
        citation="Product scope: California and Florida only",
        evidence_id="RULE-STATE-SUPPORTED",
        severity=RuleSeverity.ADVISORY,
        rationale="State outside supported jurisdictions; routing to informational mode.",
        action={"type": "route_market", "route_hint": "informational_out_of_jurisdiction"},
    )
    out = StatutoryEngineOutput(
        sanitized_profile=sanitized,
        triggered_rules=[rule],
        market_route_hints=["informational_out_of_jurisdiction"],
    )
    audit = make_node(
        agent="StatutoryAgent",
        summary="STATE-SUPPORTED short-circuit (state outside CA/FL); ReAct loop skipped.",
        evidence_ids=["RULE-STATE-SUPPORTED"],
        payload={"rule_ids": ["STATE-SUPPORTED"], "phase": "short_circuit"},
    )
    return {
        "sanitized_profile":     out.sanitized_profile,
        "triggered_rules":       out.triggered_rules,
        "statutory_violations":  out.statutory_violations,
        "field_treatments":      out.field_treatments,
        "required_offers":       out.required_offers,
        "required_coverages":    out.required_coverages,
        "required_forms":        out.required_forms,
        "floors":                out.floors,
        "market_route_hints":    out.market_route_hints,
        "decision_trace":        append_node(list(state.get("decision_trace", [])), audit),
    }


__all__ = ["statutory_agent_node"]
