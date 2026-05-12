"""Risk Assessment Agent (ReAct).

Uses LangGraph's ``create_react_agent`` over the hazard-tool subset.
Structured output is a list of RiskFactor records with evidence_ids.
"""

from __future__ import annotations

from typing import Any

from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from .. import prompts
from ..decision_trace import append_node, make_node
from ..llm_registry import AgentRole, get_llm
from ..schemas import RiskFactor, Severity
from ..tools import HAZARD_TOOLS
from ._constants import DEFAULT_LATLON


class RiskFactorList(BaseModel):
    factors: list[RiskFactor] = Field(default_factory=list)


def _seed_lat_lon(state: str) -> tuple[float, float]:
    return DEFAULT_LATLON.get(state.upper(), (0.0, 0.0))


def risk_node(state: dict[str, Any]) -> dict[str, Any]:
    """Run the ReAct loop over hazard tools; return structured RiskFactor list."""
    sanitized = state.get("sanitized_profile") or {}
    state_code = (sanitized.get("state") or "").upper()
    lat, lon = sanitized.get("lat"), sanitized.get("lon")
    if lat is None or lon is None:
        lat, lon = _seed_lat_lon(state_code)
    county_fips = sanitized.get("county_fips") or ("06037" if state_code == "CA" else "12086" if state_code == "FL" else None)
    has_pool = bool(sanitized.get("has_pool"))
    claims = int(sanitized.get("claims_history") or 0)
    market_route = state.get("market_route")

    llm = get_llm(AgentRole.RISK_AGENT)
    react_agent = create_react_agent(
        llm,
        tools=HAZARD_TOOLS,
        prompt=prompts.RISK_AGENT,
    )

    user_msg = (
        f"Customer profile (sanitized): state={state_code}, lat={lat}, lon={lon}, "
        f"county_fips={county_fips}, has_pool={has_pool}, claims_history={claims}, "
        f"market_route={market_route.value if market_route else 'unknown'}.\n\n"
        "Use the hazard tools to identify the per-peril risk factors. "
        "Return a final structured list of RiskFactor records (factor, severity in {low, medium, high}, "
        "rationale, evidence_ids[])."
    )

    structured_llm = llm.with_structured_output(RiskFactorList)

    try:
        agent_out = react_agent.invoke({"messages": [{"role": "user", "content": user_msg}]})
        # Take the final message and ask the structured LLM to coerce into RiskFactorList.
        last = agent_out["messages"][-1].content if agent_out.get("messages") else ""
        coerced = structured_llm.invoke(
            [
                {"role": "system", "content": "Convert this risk-assessment narrative into a RiskFactorList structured output. Preserve every evidence_id exactly."},
                {"role": "user", "content": last if isinstance(last, str) else str(last)},
            ]
        )
        factors = coerced.factors
    except Exception as exc:
        # Fallback: synthesise minimal factors from sanitized profile so the graph keeps moving.
        factors = _fallback_factors(state_code, has_pool, claims)
        node = make_node(
            agent="RiskAgent",
            summary=f"ReAct loop failed ({exc!r}); used deterministic fallback factor list.",
            evidence_ids=[],
        )
        return {
            "risk_factors": factors,
            "decision_trace": append_node(list(state.get("decision_trace", [])), node),
        }

    # Append a single audit node summarising the risk pass.
    node = make_node(
        agent="RiskAgent",
        summary=f"Identified {len(factors)} risk factor(s).",
        evidence_ids=[eid for f in factors for eid in (f.evidence_ids or [])],
        payload={"factor_names": [f.factor for f in factors]},
    )
    return {
        "risk_factors": factors,
        "decision_trace": append_node(list(state.get("decision_trace", [])), node),
    }


def _fallback_factors(state_code: str, has_pool: bool, claims: int) -> list[RiskFactor]:
    out: list[RiskFactor] = []
    if state_code == "CA":
        out.append(RiskFactor(factor="Wildfire", severity=Severity.HIGH, rationale="CA default wildfire exposure (fallback path)", evidence_ids=["FHSZ-OBJ-FALLBACK"]))
        out.append(RiskFactor(factor="Seismic", severity=Severity.HIGH, rationale="CA default seismic exposure (fallback path)", evidence_ids=["PGA-FALLBACK"]))
    if state_code == "FL":
        out.append(RiskFactor(factor="Hurricane", severity=Severity.HIGH, rationale="FL default hurricane exposure (fallback path)", evidence_ids=["HURDAT-FALLBACK"]))
        out.append(RiskFactor(factor="Flood", severity=Severity.MEDIUM, rationale="FL coastal flood baseline (fallback path)", evidence_ids=["NFHL-FALLBACK"]))
    if has_pool:
        out.append(RiskFactor(factor="Pool liability", severity=Severity.MEDIUM, rationale="On-premises pool flagged", evidence_ids=["MULT-POOL-HIGH"]))
    if claims >= 1:
        sev = Severity.HIGH if claims >= 2 else Severity.MEDIUM
        out.append(RiskFactor(factor="Claims history", severity=sev, rationale=f"{claims} prior claim(s)", evidence_ids=[f"MULT-CLAIMS-{claims if claims < 3 else '3PLUS'}"]))
    return out
