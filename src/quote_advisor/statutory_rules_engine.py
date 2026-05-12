"""StatutoryRulesEngine - the load-bearing pre-LLM gate (DEC-0005).

Pure Python. Runs **before any agent LLM call**. Reads ``data/tables/statutory_rules.json``
(14 rules) and produces a sanitised profile plus an audit trail of triggered rules.

Per DEC-0005 this is not a guardrail among others - it is a hard gate. Prohibited
fields (e.g., CA credit_score under Prop 103) are nulled out before any LLM sees
the profile. The Compliance Officer Council persona retains VETO power on
output-stage violations (see council.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import (
    CustomerProfile,
    RuleFire,
    RuleSeverity,
    StatutoryEngineOutput,
    normalise_state,
)

DEFAULT_RULES_PATH = Path(__file__).resolve().parents[2] / "data" / "tables" / "statutory_rules.json"


# =============================================================================
# Public API
# =============================================================================


def load_rules(path: Path | str = DEFAULT_RULES_PATH) -> list[dict[str, Any]]:
    """Load the rules JSON. Raises FileNotFoundError / JSONDecodeError loudly."""
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    rules = payload.get("rules", [])
    if not isinstance(rules, list):
        raise ValueError(
            f"statutory_rules.json: expected list under 'rules', got {type(rules)!r}"
        )
    return rules


def apply(
    raw_profile: dict[str, Any],
    *,
    context: dict[str, Any] | None = None,
    rules_path: Path | str = DEFAULT_RULES_PATH,
) -> StatutoryEngineOutput:
    """Apply all matching rules to the raw profile.

    Args:
        raw_profile: The customer profile JSON as a Python dict (per spec).
        context: Optional pre-resolved signals the engine cannot derive from the
            raw profile alone (e.g., ``{"in_sfha": bool, "has_mortgage": bool}``).
            Tools that derive these populate context on a second pass; rules
            that need them simply do not fire on the first pass.
        rules_path: Override for custom deployments.

    Returns:
        ``StatutoryEngineOutput`` with the sanitised profile plus an audit trail.
    """
    context = dict(context or {})

    # 1) Pydantic validation at the I/O boundary.
    profile = CustomerProfile.model_validate(raw_profile).with_derived_state()
    sanitized = profile.model_dump(exclude_none=False)

    # Merge context-supplied signals so triggers can see them.
    for key in ("has_mortgage", "in_sfha"):
        if key in context and context[key] is not None:
            sanitized[key] = context[key]

    # 2) Iterate rules deterministically.
    rules = load_rules(rules_path)
    out = StatutoryEngineOutput(sanitized_profile=sanitized)

    for rule in rules:
        trigger = rule.get("trigger", {}) or {}
        if not _match_trigger(trigger, sanitized):
            continue
        try:
            _apply_action(rule, sanitized, out)
        except Exception as exc:  # pragma: no cover - defensive
            out.statutory_violations.append(
                f"Engine failure applying rule {rule.get('rule_id', '<unknown>')}: {exc!r}"
            )
            continue

        out.triggered_rules.append(
            RuleFire(
                rule_id=rule["rule_id"],
                jurisdiction=rule.get("jurisdiction", "*"),
                citation=rule["citation"],
                evidence_id=rule["evidence_id"],
                severity=RuleSeverity(rule.get("severity", "advisory")),
                rationale=rule["rationale"],
                action=rule.get("action", {}),
            )
        )

    out.sanitized_profile = sanitized
    return out


# =============================================================================
# Internals - trigger matching
# =============================================================================


def _match_trigger(trigger: dict[str, Any], profile: dict[str, Any]) -> bool:
    """Pure-Python trigger evaluation.

    Supported predicates (all must match - implicit AND):
      - "<field>": <value>            -> profile[field] == value
      - "<field>": null               -> profile[field] is None
      - "<field>_not_in": [...]       -> profile[field] not in list
      - "<field>_in":     [...]       -> profile[field] in list
    """
    for key, expected in trigger.items():
        if key.endswith("_not_in"):
            field = key.removesuffix("_not_in")
            if profile.get(field) in (expected or []):
                return False
            continue
        if key.endswith("_in"):
            field = key.removesuffix("_in")
            if profile.get(field) not in (expected or []):
                return False
            continue

        actual = profile.get(key)
        if key == "state":
            actual = normalise_state(actual) if actual else actual

        if expected is None:
            if actual is not None:
                return False
        else:
            if actual != expected:
                return False
    return True


# =============================================================================
# Internals - action dispatch
# =============================================================================


def _apply_action(rule: dict[str, Any], sanitized: dict[str, Any], out: StatutoryEngineOutput) -> None:
    action = rule.get("action") or {}
    atype = action.get("type")

    if atype == "drop_field":
        field = action["field"]
        sanitized[field] = None
        out.field_treatments[field] = f"dropped_by_{rule['rule_id']}"
        return

    if atype == "flag_field_treatment":
        out.field_treatments[action["field"]] = action["treatment"]
        return

    if atype == "require_offer":
        out.required_offers.append(
            {
                "rule_id": rule["rule_id"],
                "evidence_id": rule["evidence_id"],
                "offer": action["offer"],
                **{k: v for k, v in action.items() if k not in {"type", "offer"}},
            }
        )
        return

    if atype == "require_coverage":
        out.required_coverages.append(action["coverage"])
        return

    if atype == "require_form":
        out.required_forms.append(action["form"])
        return

    if atype == "set_floor":
        floor: dict[str, Any] = {"rule_id": rule["rule_id"], "evidence_id": rule["evidence_id"]}
        if "value" in action:
            floor["value"] = action["value"]
        if "value_rule" in action:
            floor["value_rule"] = action["value_rule"]
        out.floors[action["field"]] = floor
        return

    if atype == "route_market":
        out.market_route_hints.append(action["route_hint"])
        return

    raise ValueError(f"Unknown action type: {atype!r}")
