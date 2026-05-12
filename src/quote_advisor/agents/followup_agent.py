"""Follow-up Explanation Agent (Self-Ask + DecisionTrace walker).

Never re-prompts upstream agents. Walks the persisted DecisionTrace, surfaces
top pricing drivers, optionally retrieves from a relevant RAG corpus, and
composes a natural-language answer with parenthetical evidence_id citations.
"""

from __future__ import annotations

import sys
from typing import Any

from rich.console import Console

from .. import prompts
from ..decision_trace import append_node, make_node, top_pricing_drivers
from ..llm_registry import AgentRole, get_llm
from ..rag.retriever import rag_retrieve

_console = Console(file=sys.stderr)


def _pick_corpus(question: str, state_code: str) -> tuple[str, str] | None:
    """Pick a RAG corpus for follow-up retrieval.

    Constrained to the 3 active corpora (ca_doi, fl_dfs, gse_lender).
    Consumer-facing and risk-data corpora (naic_consumer_guide, iii_handbook,
    fema_p312, calfire_defensible) are deferred — see README §16.
    """
    q = question.lower()
    if any(t in q for t in ("mortgage", "lender", "fannie", "freddie", "gse", "escrow")):
        return ("gse_lender", "national")
    if state_code == "CA":
        return ("ca_doi", "CA")
    if state_code == "FL":
        return ("fl_dfs", "FL")
    return None


def followup_node(state: dict[str, Any]) -> dict[str, Any]:
    """Answer the user's follow-up by walking the trace and (optionally) retrieving."""
    question = state.get("followup_question") or ""
    state_code = (state.get("sanitized_profile") or {}).get("state", "")
    trace = state.get("decision_trace", []) or []

    drivers = top_pricing_drivers(trace, k=3)
    chain_summary = ", ".join(
        f"{d.summary} (evidence: {', '.join(d.evidence_ids)})" for d in drivers
    )

    corpus_pick = _pick_corpus(question, state_code)
    rag_chunks: list[dict[str, Any]] = []
    if corpus_pick:
        try:
            corpus, jurisdiction = corpus_pick
            rag_out = rag_retrieve.invoke({
                "query": question, "corpus": corpus, "jurisdiction": jurisdiction, "top_k": 3,
            })
            rag_chunks = rag_out.get("chunks", []) or []
        except Exception:
            pass

    rag_summary = "\n".join(
        f"  - [{c.get('evidence_id')}] {c.get('text', '')[:300]}..." for c in rag_chunks[:3]
    )

    factor_chain = state.get("pricing_factor_chain") or []
    chain_text = "\n".join(
        f"  - {e.name if hasattr(e, 'name') else e.get('name')} = {e.multiplier if hasattr(e, 'multiplier') else e.get('multiplier'):.2f}x (evidence: {e.evidence_id if hasattr(e, 'evidence_id') else e.get('evidence_id')})"
        for e in factor_chain
    )

    user_payload = (
        f"Question: {question}\n\n"
        f"Top pricing drivers from DecisionTrace:\n{chain_summary or '(none)'}\n\n"
        f"Full factor chain:\n{chain_text or '(none)'}\n\n"
        f"Relevant RAG chunks:\n{rag_summary or '(none retrieved)'}\n\n"
        "Compose a concise, citation-rich answer to the user's question."
    )

    answer_text = ""
    try:
        llm = get_llm(AgentRole.FOLLOWUP_EXPLAIN)
        msg = llm.invoke([
            {"role": "system", "content": prompts.FOLLOWUP_EXPLAIN},
            {"role": "user", "content": user_payload},
        ])
        answer_text = msg.content if hasattr(msg, "content") else str(msg)
    except Exception:
        answer_text = (
            "Based on the persisted DecisionTrace, the largest premium drivers are: "
            + (chain_summary or "(no drivers recorded)")
            + ". RAG retrieval did not return citable chunks for this question."
        )

    _console.print(f"\n[bold]Follow-up answer:[/bold]\n  {answer_text}\n")

    node = make_node(
        agent="FollowupExplain",
        summary=f"Answered follow-up; corpus={corpus_pick[0] if corpus_pick else 'none'}; chunks={len(rag_chunks)}.",
        evidence_ids=[c.get("evidence_id") for c in rag_chunks if c.get("evidence_id")],
        payload={"question": question, "answer": answer_text},
    )

    return {
        "decision_trace": append_node(list(trace), node),
        "messages": [{"role": "assistant", "content": answer_text}],
    }
