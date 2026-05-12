"""Intent Classifier (Self-Ask).

Single structured-output LLM call. Routes the next graph step.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .. import prompts
from ..decision_trace import append_node, make_node
from ..llm_registry import AgentRole, get_llm
from ..schemas import IntentLabel, MutationAxis


class IntentResult(BaseModel):
    """Structured output from the Intent Classifier."""

    intent: IntentLabel
    rationale: str = Field(..., description="One-sentence Self-Ask decomposition trace")
    mutation_axes: list[MutationAxis] = Field(
        default_factory=list,
        description="Populated only when intent=counterfactual; parsed from the user's question",
    )


def intent_node(state: dict[str, Any]) -> dict[str, Any]:
    """Classify the user's input. Populates ``state['intent']`` and (for counterfactuals) ``mutation_axes``."""
    followup = state.get("followup_question")
    raw_profile = state.get("raw_profile") or {}

    # First-turn shortcut: a fresh profile with no follow-up text => new_quote.
    if raw_profile and not followup:
        node = make_node(
            agent="IntentClassifier",
            summary="Initial profile classified as new_quote (no follow-up text).",
            payload={"_reset_trace": True},   # custom reducer: clear prior runs' nodes
        )
        return {
            "intent": IntentLabel.NEW_QUOTE,
            "decision_trace": [node],          # bare list; reducer detects the marker and replaces
        }

    if not followup:
        # No profile and no question - default to out_of_scope.
        node = make_node(agent="IntentClassifier", summary="Empty input -> out_of_scope")
        return {
            "intent": IntentLabel.OUT_OF_SCOPE,
            "decision_trace": append_node(list(state.get("decision_trace", [])), node),
        }

    llm = get_llm(AgentRole.INTENT_CLASSIFIER).with_structured_output(IntentResult)
    result: IntentResult = llm.invoke(
        [
            {"role": "system", "content": prompts.INTENT_CLASSIFIER},
            {"role": "user", "content": followup},
        ]
    )
    payload: dict[str, Any] = {"mutation_axes": [m.model_dump() for m in result.mutation_axes]}
    if result.intent == IntentLabel.NEW_QUOTE:
        payload["_reset_trace"] = True   # fresh quote → clear prior trace
    node = make_node(
        agent="IntentClassifier",
        summary=f"Intent={result.intent.value}; rationale={result.rationale}",
        payload=payload,
    )
    if result.intent == IntentLabel.NEW_QUOTE:
        trace_update: list = [node]      # reducer detects marker and replaces
    else:
        trace_update = append_node(list(state.get("decision_trace", [])), node)
    update: dict[str, Any] = {
        "intent": result.intent,
        "decision_trace": trace_update,
    }
    if result.intent == IntentLabel.COUNTERFACTUAL and result.mutation_axes:
        # Stage axes for the Counterfactual agent.
        cf_state = dict(state.get("counterfactual_base_state") or {})
        cf_state["mutation_axes"] = [m.model_dump() for m in result.mutation_axes]
        update["counterfactual_base_state"] = cf_state
    return update
