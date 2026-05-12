"""Pydantic schemas for the I/O boundary.

Per DEC-0002 we use Pydantic only at boundaries (raw input -> CustomerProfile,
agent outputs, final QuoteOutput). Inside the LangGraph state we use TypedDict.

Phase 1 defines the minimum needed for the StatutoryRulesEngine; Phase 2
extends with QuoteOutput, RiskFactor, RecommendedCoverage, etc.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# =============================================================================
# State-name normalisation
# =============================================================================

_STATE_NAME_TO_CODE = {
    "california": "CA",
    "calif": "CA",
    "ca": "CA",
    "florida": "FL",
    "fla": "FL",
    "fl": "FL",
}


def normalise_state(value: str | None) -> str | None:
    """Normalise a free-text location to a 2-letter state code.

    Returns the original (uppercased) value if it does not match a known alias —
    the STATE-SUPPORTED rule will then route to informational-only mode.
    """
    if value is None:
        return None
    key = value.strip().lower()
    return _STATE_NAME_TO_CODE.get(key, value.strip().upper())


# =============================================================================
# CustomerProfile - matches the public input contract
# =============================================================================


class CustomerProfile(BaseModel):
    """Validates the raw JSON profile before any LLM sees it.

    The public input contract:

        { "age": 50, "location": "California", "home_value": 900000,
          "has_pool": true, "claims_history": 1, "credit_score": 700 }

    Profile B sets ``credit_score`` to ``null`` - Optional[int] honours that.
    """

    model_config = ConfigDict(extra="allow")

    age: int = Field(..., ge=0, le=120)
    location: str = Field(..., min_length=1)
    home_value: int = Field(..., ge=0)
    has_pool: bool
    claims_history: int = Field(..., ge=0)
    credit_score: int | None = Field(default=None, ge=300, le=850)

    # Derived / optional context fields (may be set by upstream tools)
    state: str | None = Field(default=None, description="2-letter code derived from location")
    has_mortgage: bool | None = None
    in_sfha: bool | None = None

    @field_validator("state", mode="before")
    @classmethod
    def _coerce_state(cls, v: Any) -> Any:  # noqa: ANN401
        if v is not None:
            return normalise_state(str(v))
        return v

    def with_derived_state(self) -> "CustomerProfile":
        """Return a copy with ``state`` populated from ``location``."""
        if self.state is not None:
            return self
        return self.model_copy(update={"state": normalise_state(self.location)})


# =============================================================================
# Statutory engine output
# =============================================================================


class RuleSeverity(str, Enum):
    MANDATORY = "mandatory"
    ADVISORY = "advisory"


class RuleAction(BaseModel):
    """Deterministic action a rule performs when its trigger matches."""

    model_config = ConfigDict(extra="allow")

    type: Literal[
        "drop_field",
        "flag_field_treatment",
        "require_offer",
        "require_coverage",
        "require_form",
        "set_floor",
        "route_market",
    ]


class RuleFire(BaseModel):
    """One rule firing record - appended to ``triggered_rules`` on the GraphState."""

    rule_id: str
    jurisdiction: str
    citation: str
    evidence_id: str
    severity: RuleSeverity
    rationale: str
    action: dict[str, Any]


class StatutoryEngineOutput(BaseModel):
    """The pre-LLM gate's contract with downstream agents.

    ``sanitized_profile`` has prohibited fields nulled out (e.g., CA credit).
    ``triggered_rules`` is the audit trail. ``statutory_violations`` is non-empty
    only when an action could not be applied (engine error, never customer-facing).
    """

    sanitized_profile: dict[str, Any]
    triggered_rules: list[RuleFire] = Field(default_factory=list)
    statutory_violations: list[str] = Field(default_factory=list)
    field_treatments: dict[str, str] = Field(
        default_factory=dict,
        description="Field -> treatment label (e.g., 'credit_score' -> 'neutral_1.0x')",
    )
    required_offers: list[dict[str, Any]] = Field(default_factory=list)
    required_coverages: list[str] = Field(default_factory=list)
    required_forms: list[str] = Field(default_factory=list)
    floors: dict[str, Any] = Field(default_factory=dict)
    market_route_hints: list[str] = Field(default_factory=list)


# =============================================================================
# Routing labels
# =============================================================================


class IntentLabel(str, Enum):
    """Output of the Intent Classifier (Self-Ask)."""

    NEW_QUOTE = "new_quote"
    EXPLANATION = "explanation"
    COUNTERFACTUAL = "counterfactual"
    OUT_OF_SCOPE = "out_of_scope"


class EligibilityRoute(str, Enum):
    """Output of Eligibility Triage (Tree-of-Thoughts)."""

    ADMITTED = "admitted"
    FAIR_DIC = "fair_dic"
    CITIZENS = "citizens"
    SURPLUS_LINES = "surplus_lines"
    INFORMATIONAL = "informational"  # out-of-jurisdiction fallback


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# =============================================================================
# Per-agent output contracts
# =============================================================================


class RiskFactor(BaseModel):
    """One element of QuoteOutput.risk_factors[].

    Adds ``evidence_ids`` for internal trace; the public output contract
    strips them down to {factor, severity, rationale} at the boundary.
    """

    factor: str
    severity: Severity
    rationale: str
    evidence_ids: list[str] = Field(default_factory=list)
    sub_score: float | None = Field(default=None, ge=0.0, le=1.0)


class RecommendedCoverage(BaseModel):
    """One element of QuoteOutput.recommended_coverages[].

    ``limit`` is a string per the public output contract (e.g., "300000",
    "100% of dwelling", "15% of Coverage A"). ISO/iso_code is internal.
    """

    type: str
    limit: str
    rationale: str
    iso_code: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class PremiumRange(BaseModel):
    low: float = Field(..., ge=0.0)
    high: float = Field(..., ge=0.0)
    currency: Literal["USD"] = "USD"

    @field_validator("high")
    @classmethod
    def _high_ge_low(cls, v: float, info) -> float:  # noqa: ANN001
        low = info.data.get("low")
        if low is not None and v < low:
            raise ValueError("PremiumRange.high must be >= low")
        return v


class FactorChainEntry(BaseModel):
    """One row in the pricing factor chain.

    Matches the ReWOO solver's chain composition:
        base * mult_1 * mult_2 * ... = point_estimate.
    """

    name: str
    multiplier: float
    evidence_id: str
    rationale: str | None = None


class PersonaName(str, Enum):
    UNDERWRITER = "conservative_underwriter"
    ADVOCATE = "customer_advocate"
    ACTUARY = "actuarial_analyst"
    COMPLIANCE = "compliance_officer"


class PersonaPosition(BaseModel):
    """Council persona output (Round 1 / Round 2)."""

    persona: PersonaName
    weight: float = Field(..., ge=0.0)
    position_summary: str
    proposed_premium_range: PremiumRange | None = None
    citations: list[str] = Field(default_factory=list)
    veto: bool = False  # only Compliance Officer may set True
    holds_position: bool = True  # set by Round 2 update


class CouncilVerdict(BaseModel):
    invoked: bool
    consensus_kind: Literal["consensus", "majority", "split", "compliance_veto", "all_disagree"] | None = None
    final_premium_range: PremiumRange | None = None
    warnings: list[str] = Field(default_factory=list)
    refer_to_human: bool = False
    positions: list[PersonaPosition] = Field(default_factory=list)


class DecisionNode(BaseModel):
    """One node of the DecisionTrace DAG.

    Every claim downstream agents make traces back to one of these nodes via
    evidence_ids. The Follow-up Agent walks this DAG instead of re-prompting.
    """

    node_id: str  # e.g., "DEC-014"
    agent: str  # source agent or component
    summary: str
    parents: list[str] = Field(default_factory=list)  # parent node_ids
    evidence_ids: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)


class CounterfactualDelta(BaseModel):
    """Output of the Counterfactual agent."""

    base_premium: PremiumRange
    cf_premium: PremiumRange
    delta_low_usd: float
    delta_high_usd: float
    delta_low_pct: float
    delta_high_pct: float
    drivers_changed: list[str] = Field(default_factory=list)
    mutations: dict[str, Any] = Field(default_factory=dict)  # field -> new value
    reflexion_notes: list[str] = Field(default_factory=list)
    plausibility_status: Literal["plausible", "refused"] = "plausible"


class ConfidenceBreakdown(BaseModel):
    overall: float = Field(..., ge=0.0, le=1.0)
    risk: float | None = None
    coverage: float | None = None
    pricing: float | None = None
    grounding: float | None = None
    council_invoked: bool = False
    council_verdict: str | None = None
    rationale_summary: str | None = Field(
        default=None,
        description="LLM-written 2-3 sentence paragraph (DEC-0012). Augmentative only; "
                    "the overall number is unaffected by it.",
    )


# =============================================================================
# QuoteOutput - the public output contract
# =============================================================================


class _RiskFactorOut(BaseModel):
    """Stripped representation matching the public output contract."""

    factor: str
    severity: Severity
    rationale: str


class _RecommendedCoverageOut(BaseModel):
    """Stripped representation matching the public output contract."""

    type: str
    limit: str
    rationale: str


class QuoteOutput(BaseModel):
    """Final structured output - the public contract for downstream consumers.

    The contract:

        {
          "risk_factors":           [ {factor, severity, rationale} ],
          "recommended_coverages":  [ {type, limit, rationale} ],
          "premium_range":          {low, high, currency: "USD"},
          "explanation":            "string",
          "confidence_score":       0.0 - 1.0,
          "warnings":               ["string"]
        }

    Extension fields (``confidence_breakdown``, ``decision_trace``, etc.) are
    surfaced only with ``--verbose``; the contract-required keys are always
    present at the top level.
    """

    model_config = ConfigDict(extra="forbid")

    risk_factors: list[_RiskFactorOut]
    recommended_coverages: list[_RecommendedCoverageOut]
    premium_range: PremiumRange
    explanation: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)

    # ---- extensions surfaced via --verbose -----------------------------------
    confidence_breakdown: ConfidenceBreakdown | None = None
    market_route: EligibilityRoute | None = None
    refer_to_human: bool = False
    factor_chain: list[FactorChainEntry] = Field(default_factory=list)
    triggered_rules: list[RuleFire] = Field(default_factory=list)
    counterfactual: CounterfactualDelta | None = None
    decision_trace: list[DecisionNode] = Field(default_factory=list)


# =============================================================================
# Counterfactual mutation directive (input to the agent)
# =============================================================================


class MutationAxis(BaseModel):
    """One mutation axis for the Counterfactual agent.

    ``new_value`` is a concrete union (string | bool | int | float | null)
    rather than ``Any`` so OpenAI's structured-output strict mode accepts the
    JSON schema (every field must have an explicit type). Counterfactual
    profile fields are all primitives in practice (has_pool, deductible,
    home_value, etc.).
    """

    field: str
    new_value: str | bool | int | float | None = None


class CounterfactualRequest(BaseModel):
    question: str
    axes: list[MutationAxis] = Field(default_factory=list)
