"""CoverageTaxonomyTool - free-text coverage name -> ISO code via rapidfuzz.

Reads ``data/tables/iso_coverage_taxonomy.csv``. Used by the Coverage Agent
to normalise output to the controlled vocabulary the external contract expects.
"""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from rapidfuzz import fuzz, process

from ._data import load_csv


class TaxonomyInput(BaseModel):
    free_text_name: str = Field(..., min_length=1)


class TaxonomyOutput(BaseModel):
    free_text_name: str
    iso_code: str
    canonical_name: str
    description: str
    evidence_id: str
    match_confidence: int


@tool("coverage_taxonomy", args_schema=TaxonomyInput)
def coverage_taxonomy(free_text_name: str) -> dict:
    """Map a free-text coverage name (e.g., 'home', 'liability', 'hurricane deductible') to its ISO/HO-3 code.

    Uses rapidfuzz token_sort_ratio. Raises if no candidate scores >= 60 - the Coverage Agent should retry with a different phrasing.
    """
    rows = load_csv("iso_coverage_taxonomy.csv")
    aliases = [r["free_text_name"] for r in rows]
    best = process.extractOne(free_text_name, aliases, scorer=fuzz.token_sort_ratio)
    if not best or best[1] < 60:
        raise ValueError(f"coverage_taxonomy: no match for {free_text_name!r} (best score {best[1] if best else 0})")
    matched = best[0]
    row = next(r for r in rows if r["free_text_name"] == matched)
    return TaxonomyOutput(
        free_text_name=free_text_name,
        iso_code=row["iso_code"],
        canonical_name=row["canonical_name"],
        description=row["description"],
        evidence_id=row["evidence_id"],
        match_confidence=best[1],
    ).model_dump()
