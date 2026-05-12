"""DecisionTrace - the audit DAG every claim ties back to.

Every agent appends DecisionNodes to the GraphState's ``decision_trace``.
Each node carries ``evidence_ids`` linking back to tool outputs, statutory
rules, RAG citations, or pricing multipliers. The Follow-up Agent walks
this DAG instead of re-prompting upstream agents (V5 §3.8).
"""

from __future__ import annotations

import itertools
from typing import Any

from .schemas import DecisionNode

# Module-level counter used to generate stable node IDs within a run.
_NODE_COUNTER = itertools.count(1)


def reset_counter() -> None:
    """Reset the node counter (call at the start of each run if numbering matters)."""
    global _NODE_COUNTER
    _NODE_COUNTER = itertools.count(1)


def make_node(
    *,
    agent: str,
    summary: str,
    parents: list[str] | None = None,
    evidence_ids: list[str] | None = None,
    payload: dict[str, Any] | None = None,
) -> DecisionNode:
    """Build a DecisionNode with an auto-incrementing ID."""
    n = next(_NODE_COUNTER)
    return DecisionNode(
        node_id=f"DEC-{n:03d}",
        agent=agent,
        summary=summary,
        parents=list(parents or []),
        evidence_ids=list(evidence_ids or []),
        payload=dict(payload or {}),
    )


def append_node(_trace_unused: list[DecisionNode] | None, node: DecisionNode) -> list[DecisionNode]:
    """Return ``[node]`` - the partial-update fragment for the operator.add-reduced trace.

    GraphState.decision_trace uses an ``operator.add`` reducer so each node's
    partial update is concatenated with the existing trace. We return a list
    of length 1 (the new node) and let the reducer handle accumulation.

    The first argument is retained for backward-compat at call sites; it is
    ignored.
    """
    return [node]


# =============================================================================
# DAG walker - used by the Follow-up Agent
# =============================================================================


def filter_by_agent(trace: list[DecisionNode], agent_name: str) -> list[DecisionNode]:
    """Return only nodes from a given agent (e.g., 'PricingAgent')."""
    return [n for n in trace if n.agent == agent_name]


def collect_evidence_ids(trace: list[DecisionNode]) -> list[str]:
    """Return the de-duplicated list of evidence_ids referenced across the trace."""
    seen: set[str] = set()
    out: list[str] = []
    for n in trace:
        for eid in n.evidence_ids:
            if eid in seen:
                continue
            seen.add(eid)
            out.append(eid)
    return out


def top_pricing_drivers(trace: list[DecisionNode], k: int = 3) -> list[DecisionNode]:
    """Return the top-k pricing-related nodes by largest multiplier.

    Each pricing node is expected to carry ``payload['multiplier']`` (float).
    Used by the Follow-up Agent to answer 'why is this expensive?'.
    """
    pricing_nodes = [n for n in trace if n.agent == "PricingAgent" and "multiplier" in n.payload]
    pricing_nodes.sort(key=lambda n: float(n.payload.get("multiplier", 1.0)), reverse=True)
    return pricing_nodes[:k]


def render_summary(trace: list[DecisionNode]) -> str:
    """Compact human-readable summary of a trace (for --verbose output)."""
    lines: list[str] = []
    for n in trace:
        cite = f" [{', '.join(n.evidence_ids)}]" if n.evidence_ids else ""
        lines.append(f"{n.node_id}  {n.agent:18}  {n.summary}{cite}")
    return "\n".join(lines)
