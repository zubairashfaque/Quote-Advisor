"""RAGRetrievalTool - hybrid (BM25 + dense) retrieval with mandatory jurisdiction filter.

Per DEC-0003 the jurisdiction metadata filter is mandatory at retrieval time
to prevent CA/FL contamination; the @tool signature makes ``jurisdiction``
required (not optional with a default) so the contract is explicit.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .store import CORPUS_JURISDICTION, CORPUS_NAMES, get_collection


class RAGRetrieveInput(BaseModel):
    query: str = Field(..., min_length=1)
    # Active corpora in v1.0. The 4 deferred corpora (naic_consumer_guide,
    # iii_handbook, fema_p312, calfire_defensible) live under
    # data/corpora/deferred/ and are not registered — see README §17.
    corpus: Literal[
        "ca_doi",
        "fl_dfs",
        "gse_lender",
    ]
    jurisdiction: Literal["national", "CA", "FL"] = Field(
        ..., description="Mandatory metadata filter; must match the corpus's tagged jurisdiction"
    )
    top_k: int = Field(default=3, ge=1, le=10)


class RAGRetrievedChunk(BaseModel):
    text: str
    evidence_id: str
    source_url: str
    jurisdiction: str
    corpus: str
    score: float | None = None


class RAGRetrieveOutput(BaseModel):
    query: str
    corpus: str
    jurisdiction: str
    chunks: list[RAGRetrievedChunk]


@tool("rag_retrieve", args_schema=RAGRetrieveInput)
def rag_retrieve(query: str, corpus: str, jurisdiction: str, top_k: int = 3) -> dict:
    """Retrieve top-k chunks from one of the 6 corpora with mandatory jurisdiction filter.

    Jurisdiction must match the corpus's tagged jurisdiction (national / CA / FL); cross-jurisdictional
    queries (e.g., asking ca_doi with jurisdiction='FL') return an empty list rather than silently
    leaking. This is the DEC-0003 hardening.
    """
    if corpus not in CORPUS_NAMES:
        raise ValueError(f"rag_retrieve: unknown corpus={corpus!r}")
    expected = CORPUS_JURISDICTION[corpus]
    if expected != "national" and jurisdiction != expected:
        # Hard guard - never return cross-jurisdictional content.
        return RAGRetrieveOutput(
            query=query,
            corpus=corpus,
            jurisdiction=jurisdiction,
            chunks=[],
        ).model_dump()

    coll = get_collection(corpus)
    where: dict | None = None
    if jurisdiction != "national":
        where = {"jurisdiction": jurisdiction}
    docs = coll.similarity_search_with_relevance_scores(query, k=top_k, filter=where)
    chunks = [
        RAGRetrievedChunk(
            text=doc.page_content,
            evidence_id=doc.metadata.get("evidence_id", ""),
            source_url=doc.metadata.get("source_url", ""),
            jurisdiction=doc.metadata.get("jurisdiction", "national"),
            corpus=corpus,
            score=score,
        )
        for doc, score in docs
    ]
    return RAGRetrieveOutput(
        query=query,
        corpus=corpus,
        jurisdiction=jurisdiction,
        chunks=chunks,
    ).model_dump()
