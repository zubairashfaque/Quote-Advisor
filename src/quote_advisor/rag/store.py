"""ChromaDB factory.

Single persistent client, lazily constructed. Embeddings use HuggingFace's
``BAAI/bge-small-en-v1.5`` (small, CPU-friendly).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ..configuration import get_settings


@lru_cache(maxsize=1)
def get_embedding():
    """Return a langchain HuggingFaceEmbeddings instance."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_chroma_dir() -> Path:
    return get_settings().chromadb_dir


def get_collection(corpus_name: str):
    """Return the langchain Chroma vectorstore for one corpus."""
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=corpus_name,
        embedding_function=get_embedding(),
        persist_directory=str(get_chroma_dir()),
    )


CORPUS_NAMES = (
    "ca_doi",
    "fl_dfs",
    "gse_lender",
)


CORPUS_JURISDICTION = {
    "ca_doi":     "CA",
    "fl_dfs":     "FL",
    "gse_lender": "national",
}

# Deferred to v1.1 (see README §16 Future Enhancements):
#   naic_consumer_guide, iii_handbook, fema_p312, calfire_defensible
# Their corpus directories live under data/corpora/deferred/.
