"""ProfileCompletenessTool - score completeness + list missing fields.

Feeds the confidence aggregator's input_completeness signal. Profile B's
``credit_score=null`` is *not* counted as missing - statutory protection,
not data quality.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

REQUIRED_FIELDS = {"age", "location", "home_value", "has_pool", "claims_history"}
STATUTORILY_OPTIONAL = {"credit_score"}  # null is allowed (FL §626.9741) and prohibited (CA Prop 103)


class ProfileCompletenessInput(BaseModel):
    profile: dict[str, Any]


class ProfileCompletenessOutput(BaseModel):
    completeness_score: float
    total_required: int
    present: int
    missing_fields: list[str]
    statutorily_optional_absent: list[str]


@tool("profile_completeness", args_schema=ProfileCompletenessInput)
def profile_completeness(profile: dict[str, Any]) -> dict:
    """Return completeness as fraction of required fields present plus the list of missing ones.

    ``credit_score`` being null is reported separately under ``statutorily_optional_absent`` so the confidence aggregator can dock it more lightly than a true missing field.
    """
    present_required = [k for k in REQUIRED_FIELDS if k in profile and profile[k] is not None]
    missing = sorted(REQUIRED_FIELDS - set(present_required))
    optional_absent = [k for k in STATUTORILY_OPTIONAL if k not in profile or profile.get(k) is None]
    score = len(present_required) / len(REQUIRED_FIELDS) if REQUIRED_FIELDS else 1.0
    return ProfileCompletenessOutput(
        completeness_score=round(score, 3),
        total_required=len(REQUIRED_FIELDS),
        present=len(present_required),
        missing_fields=missing,
        statutorily_optional_absent=optional_absent,
    ).model_dump()
