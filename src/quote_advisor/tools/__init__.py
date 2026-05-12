"""Tool registry.

Every tool in this package is a deterministic, Pydantic-IO ``@tool``-decorated
callable. They are bound to LLMs via ``llm.bind_tools(ALL_TOOLS)`` and
dispatched via LangGraph's ``ToolNode(ALL_TOOLS, handle_tool_errors=True)``.
"""

from __future__ import annotations

from .base_premium import base_premium
from .ca_fire_zone import ca_fire_zone
from .cea_recommender import cea_earthquake_recommender
from .citizens_benchmark import citizens_benchmark
from .cohort_benchmark import cohort_benchmark
from .coverage_rules import coverage_rules
from .coverage_taxonomy import coverage_taxonomy
from .credit_imputer import credit_score_imputer
from .dog_breed_liability import dog_breed_liability
from .fema_nri import fema_nri_risk
from .fl_hurricane_deductible import fl_hurricane_deductible
from .flood_zone import flood_zone
from .lender_floor import lender_floor
from .noaa_hurricane import noaa_hurricane
from .pricing_heuristic import (
    compose_premium_chain,
    home_value_scaling_factor,
    pricing_multiplier_lookup,
)
from .profile_completeness import profile_completeness
from .replacement_cost import replacement_cost
from .risk_factor import risk_factor_breakdown
from .schema_validator import schema_validator
from .state_diff import state_diff
from .usgs_seismic import usgs_seismic
from .wind_mitigation import wind_mitigation_discount

# ---- Toolset groupings (for selective bind_tools per agent) -----------------

HAZARD_TOOLS = [
    fema_nri_risk,
    ca_fire_zone,
    flood_zone,
    usgs_seismic,
    noaa_hurricane,
    dog_breed_liability,
    risk_factor_breakdown,
]

PRICING_TOOLS = [
    base_premium,
    pricing_multiplier_lookup,
    home_value_scaling_factor,
    compose_premium_chain,
    citizens_benchmark,
    cohort_benchmark,
]

COVERAGE_TOOLS = [
    coverage_rules,
    coverage_taxonomy,
    replacement_cost,
    cea_earthquake_recommender,
    fl_hurricane_deductible,
    wind_mitigation_discount,
    lender_floor,
]

STATUTORY_TOOLS = [
    credit_score_imputer,
    lender_floor,
]

UTILITY_TOOLS = [
    schema_validator,
    profile_completeness,
    state_diff,
]

# RAG retriever tool is exported from rag/retriever.py once Phase 5 is wired.

ALL_TOOLS = list(
    {
        t.name: t
        for t in (
            *HAZARD_TOOLS,
            *PRICING_TOOLS,
            *COVERAGE_TOOLS,
            *STATUTORY_TOOLS,
            *UTILITY_TOOLS,
        )
    }.values()
)


__all__ = [
    "ALL_TOOLS",
    "HAZARD_TOOLS",
    "PRICING_TOOLS",
    "COVERAGE_TOOLS",
    "STATUTORY_TOOLS",
    "UTILITY_TOOLS",
    # individual tools
    "base_premium",
    "ca_fire_zone",
    "cea_earthquake_recommender",
    "citizens_benchmark",
    "cohort_benchmark",
    "compose_premium_chain",
    "coverage_rules",
    "coverage_taxonomy",
    "credit_score_imputer",
    "dog_breed_liability",
    "fema_nri_risk",
    "fl_hurricane_deductible",
    "flood_zone",
    "home_value_scaling_factor",
    "lender_floor",
    "noaa_hurricane",
    "pricing_multiplier_lookup",
    "profile_completeness",
    "replacement_cost",
    "risk_factor_breakdown",
    "schema_validator",
    "state_diff",
    "usgs_seismic",
    "wind_mitigation_discount",
]
