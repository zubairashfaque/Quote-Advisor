"""CreditScoreImputerTool - resolve credit treatment per state.

Implements:
- CA: Prop 103 - DROP credit_score (treated as 1.0x neutral, never used as a rating factor)
- FL: Fla. Stat. §626.9741(7) - if credit_score is null, apply NEUTRAL 1.0x
- Other states: passthrough banding
"""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class CreditImputerInput(BaseModel):
    state: str = Field(..., min_length=2, max_length=2)
    credit_score: int | None = Field(default=None, ge=300, le=850)


class CreditImputerOutput(BaseModel):
    state: str
    raw_credit_score: int | None
    treatment: Literal["dropped_ca_prop103", "neutral_fl_626_9741", "excellent", "good", "fair", "poor", "unknown_pass"]
    multiplier_key: str
    citation: str
    evidence_id: str


def _band(score: int) -> Literal["excellent", "good", "fair", "poor"]:
    if score >= 760:
        return "excellent"
    if score >= 700:
        return "good"
    if score >= 640:
        return "fair"
    return "poor"


@tool("credit_score_imputer", args_schema=CreditImputerInput)
def credit_score_imputer(state: str, credit_score: int | None = None) -> dict:
    """Resolve the per-state credit treatment.

    CA: returns ``treatment='dropped_ca_prop103'`` regardless of input (Prop 103 prohibition).
    FL with null credit: returns ``treatment='neutral_fl_626_9741'`` (statutory neutral 1.0x).
    Other: bands the score into excellent / good / fair / poor.
    """
    state = state.upper()
    if state == "CA":
        return CreditImputerOutput(
            state=state,
            raw_credit_score=credit_score,
            treatment="dropped_ca_prop103",
            multiplier_key="dropped",
            citation="Cal. Code Regs. tit. 10, §2632.5 (Prop 103)",
            evidence_id="RULE-CA-PROP103-CREDIT",
        ).model_dump()
    if state == "FL" and credit_score is None:
        return CreditImputerOutput(
            state=state,
            raw_credit_score=None,
            treatment="neutral_fl_626_9741",
            multiplier_key="neutral_1.0x",
            citation="Fla. Stat. §626.9741(7)",
            evidence_id="RULE-FL-CREDIT-NEUTRAL",
        ).model_dump()
    if credit_score is None:
        return CreditImputerOutput(
            state=state,
            raw_credit_score=None,
            treatment="unknown_pass",
            multiplier_key="neutral_1.0x",
            citation="default policy when state has no codified treatment",
            evidence_id="RULE-DEFAULT-NEUTRAL",
        ).model_dump()
    band = _band(credit_score)
    return CreditImputerOutput(
        state=state,
        raw_credit_score=credit_score,
        treatment=band,
        multiplier_key=band,
        citation="industry banding (no per-state statutory override)",
        evidence_id=f"MULT-CREDIT-{band.upper()}",
    ).model_dump()
