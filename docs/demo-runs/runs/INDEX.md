# Live runs — captured 2026-05-12

All six canonical scenarios run end-to-end against the v1.0 system (4-signal confidence aggregator, 3 active RAG corpora, 7 active DECs). Each scenario writes:

- `output.json` — the structured QuoteOutput dumped by the CLI to stdout
- `stderr.log` — the `--verbose` DecisionTrace, statutory-agent ReAct trajectory, confidence breakdown, and LangSmith URL hint
- `langsmith_url.txt` — the LangSmith trace URL filtered by `thread_id`

## Summary

| Scenario | Profile | Thread | Premium range (USD) | Confidence | Counterfactual Δ | LangSmith |
|---|---|---|---|---:|---|---|
| `profile-a-new-quote` | A · CA / $900K / pool / 1 claim / credit 700 | `demo-a` | $4,790 – $7,984 | 0.95 | — | [link](profile-a-new-quote/langsmith_url.txt) |
| `profile-a-explain` | A | `demo-a` | (carry) | 0.95 | — | [link](profile-a-explain/langsmith_url.txt) |
| `profile-a-counterfactual` | A | `demo-a-v3` | (carry) | 0.95 | **−$513 to −$855 (−10.7%)** | [link](profile-a-counterfactual/langsmith_url.txt) |
| `profile-b-new-quote` | B · FL / $450K / no pool / 0 claims / credit null | `demo-b` | $5,850 – $9,750 | 0.95 | — | [link](profile-b-new-quote/langsmith_url.txt) |
| `profile-b-explain` | B | `demo-b` | (carry) | 0.95 | — | [link](profile-b-explain/langsmith_url.txt) |
| `profile-b-counterfactual` | B | `demo-b-v4` | (carry) | 0.93 | **$0 (Profile B has no pool to remove)** | [link](profile-b-counterfactual/langsmith_url.txt) |

LangSmith URLs all point at `https://smith.langchain.com/projects/refocusai/traces?metadata=thread_id:demo-{a,b}`.

## Per-scenario detail

### Profile A · new quote
- **Factor chain** (from `explanation`): `1.00 (Base CA 2026) × 2.17 (Home-value scaling) × 1.33 (claims=1) × 1.12 (pool=true) × 1.00 (credit_score dropped per Prop 103)`
- **Statutory rules applied**: `CA-PROP103-CREDIT`, `CA-AGE-NON-PRIMARY`, `CA-EQ-OFFER`, `CA-COVD-MIN-24MO`, `CA-STDFORM-2071`, `CA-FAIRPLAN-CHECK`
- **6 risk factors**, **6 recommended coverages**, **0 warnings**

### Profile A · explain follow-up
- **Question**: `"Why is this quote expensive?"`
- **FollowupAgent** retrieved 3 chunks from `ca_doi` (evidence_ids: `CDI-FAIR-01`, `CDI-STDFORM-01`, `CDI-COVD-01`)
- **DecisionTrace** walked, top pricing drivers surfaced from persisted state — no upstream agent re-prompted

### Profile A · counterfactual follow-up
- **Question**: `"What if I removed the pool?"`
- **Counterfactual agent** forked state with `mutations={'has_pool': False}`
- **Re-ran**: Risk → Coverage → Pricing on the fork (StatutoryAgent not re-run; route held constant)
- **Delta**: `delta_low = −$513 USD (−10.7%)`, `delta_high = −$855 USD (−10.7%)`, `status = plausible`
- **Captured answer text** (`followup_answer.txt`):

> If your property did not have a pool, your insurance premium would decrease by $513 on the low end and $855 on the high end, reflecting a 10.7% reduction. Originally, your premium ranged from $4,790 to $7,984, but without a pool, it would range from $4,277 to $7,129. This scenario is considered plausible, as it involves changes to the premium range, pricing factor chain, and risk factors.

### Profile B · new quote
- **Factor chain**: `1.00 (Base FL 2026) × 1.36 (Home-value scaling) × 1.00 (claims=0) × 1.00 (credit_score neutral_1.0x per §626.9741)`
- **The marquee edge case fires**: `credit_score = null` → FL `FL-CREDIT-NEUTRAL` rule applies neutral 1.0× multiplier (not worst-case 1.5×).
- **Statutory rules applied**: `FL-CREDIT-NEUTRAL`, `FL-HURRICANE-DEDUCTIBLE`, `FL-CGCC-MANDATORY`, `FL-SINKHOLE-OPTIONAL`, `FL-WIND-MITIGATION`
- **3 risk factors**, **10 recommended coverages** (includes mandatory CGCC + wind-mitigation), **0 warnings**

### Profile B · explain follow-up
- **Question**: `"Why is this quote expensive?"`
- **FollowupAgent** retrieved 3 chunks from `fl_dfs` (state-aware corpus pick)

### Profile B · counterfactual follow-up
- **Question**: `"What if I removed the pool?"`
- **Counterfactual delta**: `$0` — Profile B already has `has_pool: false`, so the mutation is a no-op. Documented honestly rather than synthesised.
- **Confidence**: 0.93 (dipped slightly from 0.95 — one Validator hard-check flagged the no-op mutation as informational).
- **Captured answer text** (`followup_answer.txt`):

> This is a no-op for your profile — you already have `has_pool=False`, so removing/changing it doesn't apply. Your premium stays at $5,850–$9,750. The premium is driven by other factors (state base rate, home-value scaling, claims history, and statutory rules), not by the field you asked to change. Status: plausible.

## How to reproduce

```bash
# Reindex first (the v1.0 system ships 3 active RAG corpora)
poetry run python -m quote_advisor.rag.ingest

# Profile A — three turns on thread demo-a
poetry run python -m quote_advisor.cli --verbose --thread-id demo-a \
  --profile examples/profile_a.json
poetry run python -m quote_advisor.cli --verbose --thread-id demo-a \
  --followup "Why is this quote expensive?"
poetry run python -m quote_advisor.cli --verbose --thread-id demo-a \
  --followup "What if I removed the pool?"

# Profile B — three turns on thread demo-b (substitute profile_b.json above)
```

Each invocation prints the QuoteOutput JSON to stdout and the DecisionTrace + LangSmith URL to stderr.
