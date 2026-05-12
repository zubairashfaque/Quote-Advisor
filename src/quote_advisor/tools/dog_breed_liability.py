"""DogBreedLiabilityTool - liability tier and surcharge for restricted dog breeds.

Reads ``data/tables/restricted_dog_breeds.csv``. Reference table; the bundled
demo profiles do not reference dogs, but the tool ships for completeness.
"""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from rapidfuzz import fuzz, process

from ._data import load_csv


class DogBreedInput(BaseModel):
    breed: str = Field(..., min_length=1, description="Free-text breed name; matched fuzzily")


class DogBreedOutput(BaseModel):
    matched_breed: str
    liability_tier: str
    surcharge_multiplier: float
    frequently_excluded: bool
    evidence_id: str
    match_confidence: int


@tool("dog_breed_liability", args_schema=DogBreedInput)
def dog_breed_liability(breed: str) -> dict:
    """Return the liability tier and surcharge multiplier for a dog breed.

    Uses rapidfuzz token_sort_ratio to match free-text breed names against the
    industry-standard restricted breed list. Falls back to the 'Other' default
    when no match scores >= 70.
    """
    rows = load_csv("restricted_dog_breeds.csv")
    breed_names = [r["breed"] for r in rows]
    match = process.extractOne(breed, breed_names, scorer=fuzz.token_sort_ratio)
    if not match or match[1] < 70:
        default = next(r for r in rows if r["breed"].lower() == "other")
        return DogBreedOutput(
            matched_breed=default["breed"],
            liability_tier=default["liability_tier"],
            surcharge_multiplier=float(default["surcharge_multiplier"]),
            frequently_excluded=default["frequently_excluded"].lower() == "true",
            evidence_id=default["evidence_id"],
            match_confidence=match[1] if match else 0,
        ).model_dump()
    matched_name = match[0]
    row = next(r for r in rows if r["breed"] == matched_name)
    return DogBreedOutput(
        matched_breed=row["breed"],
        liability_tier=row["liability_tier"],
        surcharge_multiplier=float(row["surcharge_multiplier"]),
        frequently_excluded=row["frequently_excluded"].lower() == "true",
        evidence_id=row["evidence_id"],
        match_confidence=match[1],
    ).model_dump()
