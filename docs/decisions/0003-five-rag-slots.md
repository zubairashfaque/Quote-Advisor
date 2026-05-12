# Three active RAG corpora with mandatory jurisdiction metadata filter

**Status:** Accepted *(v1.0 ships with 3 active corpora; 4 deferred to v1.1
under `data/corpora/deferred/` — see README §17 Future Enhancements)*
**Date:** 2026-05-09 *(curated 2026-05-12)*
**Note:** Title reads "five-rag-slots" for historical reasons; the project
considered up to seven during V4→V5 deliberation, and v1.0 ships three. The
jurisdiction-filter contract is unchanged regardless of corpus count.

## Context

The StatutoryAgent and the Follow-up agent both need to retrieve statute
prose and consumer-protection content at runtime. The naïve approach — one
giant RAG corpus with everything — has a critical failure mode:
**cross-jurisdictional contamination**. A CA customer's statutory analysis
should never cite Fla. Stat. §626.9741, and vice versa. The model's
similarity search will sometimes return cross-jurisdictional matches if the
prose is structurally similar.

## Decision

We ship **three active RAG corpora**, each tagged with a single `jurisdiction`
metadata value (`CA`, `FL`, or `national`), and the `rag_retrieve` tool
enforces a **mandatory jurisdiction filter** at retrieval time:

| Corpus | Jurisdiction | Content |
|---|---|---|
| `ca_doi` | CA | California Dept of Insurance — Prop 103, FAIR Plan, §2051.5, §2071 |
| `fl_dfs` | FL | Florida DFS — §626.9741, §627.701, §627.706, OIR-B1-1802 |
| `gse_lender` | national | Fannie Mae Selling Guide B7-3-02 (mortgage floor) |

**Deferred to v1.1** (live under `data/corpora/deferred/`): `naic_consumer_guide`,
`iii_handbook`, `fema_p312`, `calfire_defensible`. These augment follow-up
explanations but are not on either demo profile's gate path. See README §17.

The tool signature makes `jurisdiction` **required** (Pydantic
`Literal["national", "CA", "FL"]` with no default). The retriever applies
two defenses:
1. **Python-level hard block** at `rag/retriever.py:62-69` — querying a CA
   corpus with `jurisdiction='FL'` returns `[]` immediately, never reaches
   ChromaDB.
2. **Chroma `where={"jurisdiction": ...}`** filter — defense in depth at the
   vector-store level.

## Consequences

**Positive**
- Cross-jurisdictional contamination is structurally impossible at retrieval.
- The mandatory filter is documented in the prompt; the StatutoryAgent's
  ReAct trajectory shows the jurisdiction value on every call.
- The Phase-4 self-check in the StatutoryAgent verifies every emitted rule's
  evidence_id traces back to a chunk that was actually retrieved during that
  run.

**Negative**
- Adding a new state requires a new corpus, new prompt-routing rules, and
  new evidence_id naming patterns. Not zero-effort.
- Three corpora keeps index storage and ingest time small; total file count
  is ~9 markdown source files, ~30 ChromaDB chunks. The deferred 4 corpora
  remain on disk and can be re-activated by moving them out of
  `data/corpora/deferred/` and re-running `make ingest`.

## Alternatives considered

- Single corpus with jurisdiction as a metadata field — rejected because
  similarity-search can still surface cross-jurisdictional matches even with
  a `where` filter if the prose embedding is too similar. Defense-in-depth
  requires the Python-level hard block too.
- Three corpora (CA, FL, national) — too coarse; the regulator-specific
  corpora and the consumer-guide corpora serve different consumer agents.
