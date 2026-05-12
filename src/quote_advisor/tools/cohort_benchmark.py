"""CohortBenchmarkTool - p10 / p50 / p90 premium band by state x value x peril tier.

Reads ``data/tables/claims_cohort_benchmarks.csv``. Synthetic-but-calibrated;
flagged in data/REAL_DATA_PROVENANCE.md and README §14. Used by the Validator
to challenge pricing outliers and by the Counterfactual to gauge plausibility.
"""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ._data import load_csv


class CohortInput(BaseModel):
    state: str = Field(..., min_length=2, max_length=2)
    home_value_usd: float = Field(..., ge=0.0)
    hurricane_tier: str | None = Field(default=None, description="Required for FL; ignored for CA")
    wildfire_tier: str | None = Field(default=None, description="Required for CA; ignored for FL")


class CohortOutput(BaseModel):
    state: str
    value_band: str
    hurricane_tier: str
    wildfire_tier: str
    p10_premium_usd: float
    p50_premium_usd: float
    p90_premium_usd: float
    n_synthetic: int
    evidence_id: str


def _value_band(home_value_usd: float) -> str:
    if home_value_usd < 250_000:
        return "100-250K"
    if home_value_usd < 500_000:
        return "250-500K"
    if home_value_usd < 750_000:
        return "500-750K"
    if home_value_usd < 1_000_000:
        return "750-1000K"
    return "1000K+"


@tool("cohort_benchmark", args_schema=CohortInput)
def cohort_benchmark(
    state: str,
    home_value_usd: float,
    hurricane_tier: str | None = None,
    wildfire_tier: str | None = None,
) -> dict:
    """Return the p10/p50/p90 cohort premium band for a state x dwelling-value x peril-tier slice.

    Used by the Validator: a Pricing output above p90 triggers Council escalation; below p10 triggers a low-confidence warning.
    """
    rows = load_csv("claims_cohort_benchmarks.csv")
    state = state.upper()
    band = _value_band(home_value_usd)

    if state == "FL":
        tier_match = (hurricane_tier or "Very High").strip()
        candidates = [
            r for r in rows
            if r["state"] == state and r["value_band"] == band and r["hurricane_tier"] == tier_match
        ]
    elif state == "CA":
        tier_match = (wildfire_tier or "Moderate").strip()
        candidates = [
            r for r in rows
            if r["state"] == state and r["value_band"] == band and r["wildfire_tier"] == tier_match
        ]
    else:
        candidates = [r for r in rows if r["state"] == state and r["value_band"] == band]

    if not candidates:
        # Soft fall-back: ignore tier, take the value band match for the state.
        candidates = [r for r in rows if r["state"] == state and r["value_band"] == band]
    if not candidates:
        raise ValueError(f"cohort_benchmark: no rows for state={state} value_band={band}")

    row = candidates[0]
    return CohortOutput(
        state=row["state"],
        value_band=row["value_band"],
        hurricane_tier=row.get("hurricane_tier", "N/A"),
        wildfire_tier=row.get("wildfire_tier", "N/A"),
        p10_premium_usd=float(row["p10_premium"]),
        p50_premium_usd=float(row["p50_premium"]),
        p90_premium_usd=float(row["p90_premium"]),
        n_synthetic=int(row["n_synthetic"]),
        evidence_id=row["evidence_id"],
    ).model_dump()
