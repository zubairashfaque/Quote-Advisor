"""RAG layer - 6 corpora indexed into ChromaDB with mandatory jurisdiction filter.

Public surface:
    - ``rag_retrieve``  : the @tool used by Coverage / Validator / Follow-up agents
    - ``ingest.main()`` : offline indexer (`make ingest` / `python -m quote_advisor.rag.ingest`)
"""

from .retriever import rag_retrieve

__all__ = ["rag_retrieve"]
