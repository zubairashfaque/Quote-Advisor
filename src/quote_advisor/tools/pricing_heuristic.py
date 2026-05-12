"""PricingHeuristicTool - look up multipliers and compose a chain.

Reads ``data/tables/pricing_multipliers.json``. Pure-Python lookup + chain
arithmetic; no LLM. Used by the Pricing Agent's ReWOO Workers (each Worker
fans out a single multiplier lookup) and by the Solver as a sanity composer.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ._data import load_json


class MultiplierLookupInput(BaseModel):
    dimension: Literal[
        "wildfire", "seismic", "hurricane", "flood",
        "claims", "pool", "credit_score",
    ]
    key: str = Field(..., description="Tier label or treatment key (e.g., 'Very High', 'neutral_1.0x', '0', '1', 'true')")


class MultiplierLookupOutput(BaseModel):
    dimension: str
    key: str
    multiplier: float
    evidence_id: str


@tool("pricing_multiplier_lookup", args_schema=MultiplierLookupInput)
def pricing_multiplier_lookup(dimension: str, key: str) -> dict:
    """Look up one pricing multiplier from the canonical multipliers table.

    ReWOO Pricing Workers call this once per multiplier dimension in parallel; the Solver composes the chain.
    Returns ``multiplier`` (e.g., 2.0 for wildfire 'Very High') and the ``evidence_id`` to cite in the factor_chain.
    """
    table = load_json("pricing_multipliers.json")
    bucket = table.get(dimension)
    if not isinstance(bucket, dict):
        raise ValueError(f"pricing_multiplier_lookup: unknown dimension={dimension!r}")
    entry = bucket.get(str(key))
    if entry is None:
        raise ValueError(f"pricing_multiplier_lookup: unknown key={key!r} in dimension={dimension}")
    return MultiplierLookupOutput(
        dimension=dimension,
        key=str(key),
        multiplier=float(entry["value"]),
        evidence_id=entry["evidence_id"],
    ).model_dump()


class ScalingFactorInput(BaseModel):
    home_value_usd: float = Field(..., ge=0.0)
    base_dwelling_usd: float = Field(default=250_000.0, ge=0.0)


class ScalingFactorOutput(BaseModel):
    home_value_usd: float
    scaling_multiplier: float
    evidence_id: str
    explanation: str


@tool("home_value_scaling_factor", args_schema=ScalingFactorInput)
def home_value_scaling_factor(home_value_usd: float, base_dwelling_usd: float = 250_000.0) -> dict:
    """Return the multiplicative scaling factor that adjusts the base premium for the actual dwelling value.

    Per the multipliers table: each $100K above the $250K reference adds 18% to base. Linear interpolation; floors at 1.0x for values <= base.
    """
    table = load_json("pricing_multipliers.json")
    per_100k = float(table["home_value_scaling"]["per_100k_above_250k"]["value"])
    evidence_id = table["home_value_scaling"]["per_100k_above_250k"]["evidence_id"]
    if home_value_usd <= base_dwelling_usd:
        mult = 1.0
        explanation = f"home value {home_value_usd:.0f} <= base {base_dwelling_usd:.0f}; no scaling"
    else:
        excess_100k = (home_value_usd - base_dwelling_usd) / 100_000.0
        mult = 1.0 + per_100k * excess_100k
        explanation = f"({home_value_usd:.0f} - {base_dwelling_usd:.0f}) / 100K = {excess_100k:.2f} units * {per_100k} = {mult-1:.2f} added to 1.0"
    return ScalingFactorOutput(
        home_value_usd=home_value_usd,
        scaling_multiplier=round(mult, 4),
        evidence_id=evidence_id,
        explanation=explanation,
    ).model_dump()


class ComposeChainInput(BaseModel):
    base_premium_usd: float = Field(..., ge=0.0)
    multipliers: list[float] = Field(..., min_length=1)


class ComposeChainOutput(BaseModel):
    base_premium_usd: float
    multipliers: list[float]
    point_estimate_usd: float
    range_low_usd: float
    range_high_usd: float


@tool("compose_premium_chain", args_schema=ComposeChainInput)
def compose_premium_chain(base_premium_usd: float, multipliers: list[float]) -> dict:
    """Compose a ReWOO factor chain into a point estimate plus +/-25% range.

    Pure arithmetic - no LLM. The Solver typically returns the prose; this tool produces the numbers.
    """
    point = base_premium_usd
    for m in multipliers:
        point *= m
    return ComposeChainOutput(
        base_premium_usd=base_premium_usd,
        multipliers=multipliers,
        point_estimate_usd=round(point, 2),
        range_low_usd=round(point * 0.75, 2),
        range_high_usd=round(point * 1.25, 2),
    ).model_dump()
