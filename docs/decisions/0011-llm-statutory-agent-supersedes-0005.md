# DEC-0011 · LLM Statutory Agent (ReAct + RAG) supersedes the deterministic engine

**Status:** Accepted *(supersedes DEC-0005; the deterministic engine is kept
as the Phase-5 safety-net fallback)*
**Date:** 2026-05-10

## Context

DEC-0005 placed a hardcoded `statutory_rules_engine.py` at the gate position
in the pipeline. It worked, but two problems became material:

1. **Statute updates required code changes.** Adding a new CA bulletin or
   a FL OIR amendment meant editing Python, opening a PR, releasing. The
   rules engine encodes statutes in `if` statements; the prose lives in
   the developer's head, not in a reviewable artifact.
2. **The audit chain was weaker than it could be.** A rule like
   `CA-PROP103-CREDIT` fired because Python code said so. The
   `evidence_id` was a literal string in code, not a chunk of statute
   prose anyone could read and verify.

We needed to move the statutory ground truth out of code and into
retrievable corpora — while keeping the determinism guarantee that the
deterministic engine offered.

## Decision

The gate position is now a **5-phase ReAct + RAG StatutoryAgent**. The
legacy deterministic engine is kept as Phase 5 (safety-net fallback) and
no longer sits at the entry.

Pipeline:

| Phase | What it does | Failure mode |
|---|---|---|
| 1 · Pre-filter | Validate raw profile via `CustomerProfile`; normalize state; short-circuit `STATE-SUPPORTED` for non-CA/FL | ValidationError → Phase 5 |
| 2 · ReAct loop | `create_react_agent` with `rag_retrieve` as the only tool; 3-8 iterations; LLM picks queries based on profile signals | LLM raises → Phase 5 |
| 3 · Structured emission | Second LLM call with `with_structured_output(_StatutoryAgentEmission)` coerces narrative → typed schema | Validation error → Phase 5 |
| 4 · Self-check | Every emitted `evidence_id` must trace to a chunk that was actually retrieved during this run. Drop unmatched rules | > 50% dropped → Phase 5 |
| 5 · Safety net | Legacy `statutory_rules_engine.apply()` fires; same 8-field shape; `[FALLBACK]` audit node prepended to DecisionTrace | (terminal) |

The agent emits the **same 8-field `StatutoryEngineOutput`** downstream
agents have always consumed. Downstream code is unaware of which phase
produced the output — the contract is preserved.

The four corpus-less rules that the legacy engine handled
(`CA-STDFORM-2071`, `FL-WIND-MITIGATION`, `GSE-COV-A-FLOOR`,
`STATE-SUPPORTED`) are now backed by chunks under
`data/corpora/{ca_doi,fl_dfs,gse_lender}/` plus a `STATE-SUPPORTED`
short-circuit in Phase 1.

## Consequences

**Positive**
- **Statute updates are now corpus updates, not code changes.** Add a
  Markdown chunk with the right frontmatter to `data/corpora/ca_doi/` and
  the agent can cite it.
- **The audit chain bottoms out in the regulator's published prose.** A
  reviewer can grep an `evidence_id`, find the Markdown chunk, read its
  frontmatter `source_url`, and visit the regulator's page.
- **Safety net guarantees the pipeline never ships malformed statutory
  output.** Phase 5 is the deterministic engine that was good enough as
  the gate for DEC-0005; it's still good enough as a fallback.
- **Per-phase observability** in LangSmith. A reviewer can see which
  phase produced the output for any given run.

**Negative**
- Two LLM calls per quote (ReAct loop + structured emission), so the
  StatutoryAgent is the second-most expensive node (after Pricing's
  Planner+Solver).
- The Phase-4 self-check is critical: without it, the LLM could fabricate
  citations. The check is enforced by exact-match against the run's
  `retrieved_chunks` dict.
- The legacy engine must remain in maintenance — it's the safety net.
  Both code paths must stay in sync on the 8-field output contract.

## Verification

`tests/verify_replacements.py` is the regression contract. It runs both
demo profiles end-to-end and asserts:
- Profile A (CA) — `credit_score` is dropped; CA-PROP103-CREDIT fires.
- Profile B (FL) — `field_treatments['credit_score'] == 'neutral_1.0x'`;
  FL-CREDIT-NEUTRAL fires.
- The deterministic safety-net engine still produces statutorily-correct
  output when forced (via test seam).
- The hard-cap-at-0.5 confidence rule still fires on statutory
  non-compliance.

## Alternatives considered

- Keep DEC-0005 as is — rejected for the statute-update-requires-code-change
  pain.
- Pure LLM (no safety net) — rejected because consistency on the gate-position
  output is non-negotiable; a malformed statutory output cannot ship.
