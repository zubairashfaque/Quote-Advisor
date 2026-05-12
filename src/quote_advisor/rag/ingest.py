"""Offline RAG indexer.

Chunks the 3 active corpora under ``data/corpora/`` into ChromaDB collections
with mandatory ``jurisdiction`` metadata. Each markdown file's YAML frontmatter
contributes ``evidence_id``, ``jurisdiction``, ``source_url``, ``corpus``.

Active corpora: ca_doi, fl_dfs, gse_lender. The 4 deferred corpora
(naic_consumer_guide, iii_handbook, fema_p312, calfire_defensible) live under
``data/corpora/deferred/`` and are not indexed — see README §16.

Run as a module so ``langgraph.json`` can register it:
    python -m quote_advisor.rag.ingest

The ``graph`` symbol is a no-op LangGraph object exposed for langgraph.json
registration; the actual ingestion happens in ``main()``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..configuration import REPO_ROOT, get_settings
from .store import CORPUS_NAMES, get_collection, get_embedding


CORPORA_ROOT = REPO_ROOT / "data" / "corpora"


# =============================================================================
# Frontmatter parser (simple YAML subset; avoids pulling pyyaml just for this)
# =============================================================================


_FRONT_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    m = _FRONT_RE.match(text)
    if not m:
        return {}, text
    head, body = m.group(1), m.group(2)
    meta: dict[str, Any] = {}
    for line in head.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip("\"'")
    return meta, body


# =============================================================================
# Chunking - simple paragraph-based chunker with overlap
# =============================================================================


def chunk_markdown(body: str, *, target_chars: int = 1100, overlap_chars: int = 150) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        if current_len + len(para) > target_chars and current:
            chunks.append("\n\n".join(current))
            tail = current[-1]
            current = [tail[-overlap_chars:]] if len(tail) > overlap_chars else [tail]
            current_len = sum(len(p) for p in current)
        current.append(para)
        current_len += len(para)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


# =============================================================================
# Indexer
# =============================================================================


def index_corpus(corpus_name: str) -> int:
    corpus_dir = CORPORA_ROOT / corpus_name
    if not corpus_dir.exists():
        print(f"[ingest] {corpus_name}: directory missing ({corpus_dir}); skipping.")
        return 0

    md_files = sorted(corpus_dir.glob("*.md"))
    if not md_files:
        print(f"[ingest] {corpus_name}: no .md chunks; skipping.")
        return 0

    collection = get_collection(corpus_name)

    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []

    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)
        chunks = chunk_markdown(body)
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{corpus_name}::{md_path.stem}::{i:03d}"
            documents.append(chunk_text)
            metadatas.append(
                {
                    "corpus":       corpus_name,
                    "jurisdiction": meta.get("jurisdiction", "national"),
                    "evidence_id":  meta.get("evidence_id", chunk_id),
                    "source_url":   meta.get("source_url", ""),
                    "filename":     md_path.name,
                    "chunk_idx":    i,
                }
            )
            ids.append(chunk_id)

    if documents:
        collection.add_texts(texts=documents, metadatas=metadatas, ids=ids)
    print(f"[ingest] {corpus_name}: indexed {len(documents)} chunks from {len(md_files)} file(s).")
    return len(documents)


def main() -> int:
    settings = get_settings()
    settings.chromadb_dir.mkdir(parents=True, exist_ok=True)
    # Touch the embedding to trigger model download up-front (one-time cost).
    _ = get_embedding()

    total = 0
    for name in CORPUS_NAMES:
        total += index_corpus(name)
    print(f"\n[ingest] done; {total} chunks across {len(CORPUS_NAMES)} corpora at {settings.chromadb_dir}")
    return 0


# Stub graph object for langgraph.json registration.
# Real LangGraph apps that need an ingestion-as-a-graph wrap main() in a node;
# we keep ingest as a plain script for clarity and expose a minimal
# placeholder so langgraph.json doesn't 404 on the indexer entry.
class _IngestGraphStub:
    def invoke(self, *args: Any, **kwargs: Any) -> int:
        return main()


graph = _IngestGraphStub()


if __name__ == "__main__":
    raise SystemExit(main())
