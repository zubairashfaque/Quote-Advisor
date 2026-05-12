"""SchemaValidatorTool - structural validation against a registered Pydantic schema.

Used by the retry-with-validation guardrail (Phase 9): when an agent's
structured output does not validate, the validation errors are appended to
the next prompt so the model can self-correct.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field, ValidationError

from .. import schemas as _schemas

_REGISTRY: dict[str, type[BaseModel]] = {
    "CustomerProfile":         _schemas.CustomerProfile,
    "RuleFire":                _schemas.RuleFire,
    "StatutoryEngineOutput":   _schemas.StatutoryEngineOutput,
    "RiskFactor":              _schemas.RiskFactor,
    "RecommendedCoverage":     _schemas.RecommendedCoverage,
    "PremiumRange":            _schemas.PremiumRange,
    "FactorChainEntry":        _schemas.FactorChainEntry,
    "PersonaPosition":         _schemas.PersonaPosition,
    "CouncilVerdict":          _schemas.CouncilVerdict,
    "CounterfactualDelta":     _schemas.CounterfactualDelta,
    "CounterfactualRequest":   _schemas.CounterfactualRequest,
    "QuoteOutput":             _schemas.QuoteOutput,
}


class SchemaValidatorInput(BaseModel):
    schema_name: str = Field(..., description="One of the names in tools.schema_validator._REGISTRY")
    payload: dict[str, Any]


class SchemaValidatorOutput(BaseModel):
    valid: bool
    errors: list[dict[str, Any]] = Field(default_factory=list)
    schema_name: str


@tool("schema_validator", args_schema=SchemaValidatorInput)
def schema_validator(schema_name: str, payload: dict[str, Any]) -> dict:
    """Validate a payload against a named Pydantic schema. Returns ``valid`` + list of structured ``errors``."""
    cls = _REGISTRY.get(schema_name)
    if cls is None:
        raise ValueError(f"schema_validator: unknown schema_name={schema_name!r}; registered: {sorted(_REGISTRY)}")
    try:
        cls.model_validate(payload)
    except ValidationError as ve:
        return SchemaValidatorOutput(
            valid=False,
            errors=[
                {"loc": list(err["loc"]), "type": err["type"], "msg": err["msg"]}
                for err in ve.errors()
            ],
            schema_name=schema_name,
        ).model_dump()
    return SchemaValidatorOutput(valid=True, schema_name=schema_name).model_dump()
