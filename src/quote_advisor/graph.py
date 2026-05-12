"""LangGraph wiring (Phase 8).

build_graph() returns the compiled StateGraph with MemorySaver checkpointer.
The exported ``graph`` symbol is what ``langgraph.json`` registers.

Flow:

  Intent
    -> out_of_scope  -> refusal             -> END
    -> explanation   -> followup            -> END
    -> counterfactual-> counterfactual      -> output -> END
    -> new_quote     -> statutory_gate -> eligibility -> risk -> coverage
                       -> pricing_planner -> [Send -> pricing_worker (parallel)]
                       -> pricing_solver  -> validator
                            -> (flagged) council -> confidence -> output -> END
                            -> (clean)            confidence -> output -> END
"""

from __future__ import annotations

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from .agents import (
    council_round_node,
    counterfactual_node,
    coverage_node,
    eligibility_node,
    followup_node,
    intent_node,
    pricing_planner_node,
    pricing_solver_node,
    pricing_worker_node,
    risk_node,
    validator_node,
)
from .configuration import get_settings
from .nodes import (
    confidence_node,
    output_assembler_node,
    pricing_dispatch_router,
    refusal_node,
    route_after_validator,
    route_intent,
    statutory_gate_node,
)
from .state import GraphState


def build_graph(*, with_checkpointer: bool = True):
    """Construct and compile the multi-agent graph."""
    s = get_settings()  # bootstraps LangSmith if enabled

    builder = StateGraph(GraphState)

    # ---- Nodes ----------------------------------------------------------------
    builder.add_node("intent",            intent_node)
    builder.add_node("statutory_gate",    statutory_gate_node)
    builder.add_node("eligibility",       eligibility_node)
    builder.add_node("risk",              risk_node)
    builder.add_node("coverage",          coverage_node)
    builder.add_node("pricing_planner",   pricing_planner_node)
    builder.add_node("pricing_worker",    _wrap_worker(pricing_worker_node))
    builder.add_node("pricing_solver",    pricing_solver_node)
    builder.add_node("validator",         validator_node)
    builder.add_node("council",           council_round_node)
    builder.add_node("counterfactual",    counterfactual_node)
    builder.add_node("followup",          followup_node)
    builder.add_node("confidence",        confidence_node)
    builder.add_node("output",            output_assembler_node)
    builder.add_node("refusal",           refusal_node)

    # ---- Entry: Intent --------------------------------------------------------
    builder.add_edge(START, "intent")

    # Intent routes to the right subgraph.
    builder.add_conditional_edges(
        "intent",
        route_intent,
        {
            "statutory_gate":  "statutory_gate",
            "followup":        "followup",
            "counterfactual":  "counterfactual",
            "refusal":         "refusal",
        },
    )

    # ---- New-quote main chain -------------------------------------------------
    builder.add_edge("statutory_gate",  "eligibility")
    builder.add_edge("eligibility",     "risk")
    builder.add_edge("risk",            "coverage")
    builder.add_edge("coverage",        "pricing_planner")

    # Pricing planner -> Send-fanned workers -> solver
    builder.add_conditional_edges("pricing_planner", pricing_dispatch_router, ["pricing_worker"])
    builder.add_edge("pricing_worker",  "pricing_solver")
    builder.add_edge("pricing_solver",  "validator")

    builder.add_conditional_edges(
        "validator",
        route_after_validator,
        {"council": "council", "confidence": "confidence"},
    )
    builder.add_edge("council",         "confidence")

    # All paths converge on confidence -> output -> END
    builder.add_edge("confidence",      "output")
    builder.add_edge("counterfactual",  "confidence")
    builder.add_edge("followup",        "output")
    builder.add_edge("refusal",         END)
    builder.add_edge("output",          END)

    # ---- Compile with checkpointer -------------------------------------------
    if with_checkpointer:
        ckpt_path = str(s.checkpoint_sqlite)
        s.checkpoint_sqlite.parent.mkdir(parents=True, exist_ok=True)
        # SqliteSaver.from_conn_string is a context manager; we build at import time
        # using a directly-opened connection so the graph remains usable across calls.
        import sqlite3
        conn = sqlite3.connect(ckpt_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        return builder.compile(checkpointer=checkpointer)
    return builder.compile()


def _wrap_worker(worker_fn):
    """Worker is called with the per-task dict (not the full GraphState).

    LangGraph's Send dispatches the task payload as the node input; the worker
    returns ``{'pricing_results': [...]}`` which LangGraph reduces by
    appending lists across parallel workers (provided GraphState's reducer
    config supports it - we use ``operator.add`` for ``pricing_results`` via
    a custom reducer in state.py if needed).
    """
    return worker_fn


# Exposed for langgraph.json
graph = build_graph()
