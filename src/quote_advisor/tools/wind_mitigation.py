"""WindMitigationDiscountTool - FL OIR-B1-1802 wind premium discount calculator.

Reads ``data/tables/fl_wind_mitigation_form.json``. Discount components are
additive then capped at 45% (statutory cap).
"""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ._data import load_json


class WindMitigationInput(BaseModel):
    roof_shape: Literal["hip", "gable", "flat", "other"] = "other"
    roof_cover: Literal["FBC_compliant", "non_FBC"] = "non_FBC"
    roof_deck_attachment: Literal["A", "B", "C"] = "A"
    opening_protection: Literal["all_impact", "basic_impact", "none"] = "none"
    secondary_water_resistance: bool = False


class WindComponent(BaseModel):
    dimension: str
    selection: str
    discount_pct: float
    evidence_id: str


class WindMitigationOutput(BaseModel):
    components: list[WindComponent]
    aggregate_discount_pct: float
    capped_at: float
    citation: str
    evidence_id: str


@tool("wind_mitigation_discount", args_schema=WindMitigationInput)
def wind_mitigation_discount(
    roof_shape: str = "other",
    roof_cover: str = "non_FBC",
    roof_deck_attachment: str = "A",
    opening_protection: str = "none",
    secondary_water_resistance: bool = False,
) -> dict:
    """Compute the FL OIR-B1-1802 wind-premium discount aggregate.

    Per the form, eligible discounts are summed across roof_shape, roof_cover, roof_deck_attachment, opening_protection, and secondary_water_resistance, then capped at 45% (statutory).
    """
    table = load_json("fl_wind_mitigation_form.json")
    dims = table["discount_dimensions"]
    components: list[WindComponent] = []
    selections = {
        "roof_shape":                  roof_shape,
        "roof_cover":                  roof_cover,
        "roof_deck_attachment":        roof_deck_attachment,
        "opening_protection":          opening_protection,
        "secondary_water_resistance":  str(secondary_water_resistance).lower(),
    }
    total = 0.0
    for dim, sel in selections.items():
        block = dims[dim].get(sel) if isinstance(dims.get(dim), dict) else None
        if block is None:
            block = next(iter(dims[dim].values()))
            sel = "fallback"
        pct = float(block["discount_pct"])
        components.append(
            WindComponent(dimension=dim, selection=sel, discount_pct=pct, evidence_id=block["evidence_id"])
        )
        total += pct
    cap = float(table["max_aggregate_discount_pct"])
    return WindMitigationOutput(
        components=components,
        aggregate_discount_pct=round(min(total, cap), 4),
        capped_at=cap,
        citation=table["citation"],
        evidence_id=table["evidence_id"],
    ).model_dump()
