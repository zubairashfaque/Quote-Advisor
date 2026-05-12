# 🎬 Demo walkthroughs — agent & tool breakdowns

> 📺 **Prefer the slide deck?** Open [`slides.html`](slides.html) in any browser — single-file reveal.js presentation with all four demos visualised, charts, and embedded mermaid diagrams. No build step.

Each file in this directory documents one real demo run end-to-end:

- the cognitive pattern of every agent that fired
- the input it received, the output it produced, and **why** it produced that output
- the tools each agent invoked, the data they consulted, and the values they returned
- the final structured `QuoteOutput`
- the LangSmith trace URL for the run

All values shown are **real, captured live** from the runs (no placeholders).

---

## 📚 The five runs

| # | Doc | Demo command | What it tests | Trace files | LangSmith URL |
|---|---|---|---|---|---|
| 1 | **[walkthrough.html ▸ 05](walkthrough.html#demo-a)** | `make demo-a` | Full 14-node pipeline for a California customer (Profile A: $900K, pool, 1 claim, credit 700). Fires CA-PROP103-CREDIT (drops credit), routes `fair_dic` (high FHSZ), 6 risk factors, CEA earthquake endorsement. | `traces/01-demo-a.{stdout.json,verbose.log}` | https://smith.langchain.com/projects/refocusai/traces?metadata=thread_id:demo-a |
| 2 | **[01-demo-b.md](01-demo-b.md)** + **[walkthrough.html ▸ 06](walkthrough.html#demo-b)** | `make demo-b` | Full pipeline for a Florida customer with `credit_score=null` (Profile B). Fires FL-CREDIT-NEUTRAL per §626.9741, 3 hurricane deductible options per §627.701, CGCC mandatory per §627.706, wind-mitigation advisory per OIR-B1-1802. | `traces/02-demo-b.{stdout.json,verbose.log}` | https://smith.langchain.com/projects/refocusai/traces?metadata=thread_id:demo-b |
| 3 | **[02-followup-explain.md](02-followup-explain.md)** + **▸ 07** | `make followup-explain` | Self-Ask follow-up that walks the prior `demo-a` DecisionTrace to answer *"Why is this quote expensive?"* — **without re-running upstream agents**. | `traces/03-followup-explain.*` | demo-a |
| 4 | **[03-followup-cf.md](03-followup-cf.md)** + **▸ 08** | `make followup-cf` | Counterfactual: forks `demo-a` state with `has_pool=False`, re-runs Risk → Coverage → Pricing on the fork, returns delta `−$513/−$855 (−10.7%)`. | `traces/04-followup-cf.*` | demo-a |
| 5 | **[04-followup-cf-multi.md](04-followup-cf-multi.md)** + **▸ 09** | `make followup-cf-multi` | Multi-axis counterfactual with mutations `{has_pool: false, credit_score: 600}`. Same delta as the single-axis case — credit drop is statutorily nullified (CA-PROP103-CREDIT re-fires on the fork). | `traces/05-followup-cf-multi.*` | demo-a |

**The walkthrough** [`walkthrough.html`](walkthrough.html) is the editorial top-to-bottom view: 16 sections, 5 live runs, full 5W1H per agent in demo-a + demo-b, every CLI flag explained, every routing predicate documented.

---

## 🗺️ Master matrix — agents × demos × tools

The following matrix shows which agents and tools fired in each run.

| Agent | demo-a | demo-b | explain | cf | cf-multi | Cognitive pattern |
|---|---|---|---|---|---|---|
| 🔵 Intent Classifier      | ✅ | ✅ | ✅ | ✅ | ✅ | Self-Ask |
| 🤖 StatutoryAgent         | ✅·6 rules | ✅·5 rules | ➖ | ✅¹ | ✅¹ | ReAct + RAG (DEC-0011; legacy engine kept as Phase-5 fallback) |
| 🌳 Eligibility Triage     | ✅·fair_dic | ✅·citizens | ➖ | ✅¹ | ✅¹ | Tree-of-Thoughts |
| 🔁 Risk Assessment        | ✅·6 factors | ✅·3 factors | ➖ | ✅¹ | ✅¹ | ReAct |
| 📋 Coverage Recommendation| ✅·6 lines | ✅·10 lines | ➖ | ✅¹ | ✅¹ | Plan-and-Execute |
| ⚡ Pricing (planner)      | ✅·9 tasks | ✅·8 tasks | ➖ | ✅¹ | ✅¹ | ReWOO |
| ⚙️  Pricing workers (×N)  | ✅ | ✅ | ➖ | ✅¹ | ✅¹ | (deterministic, no LLM) |
| 🧮 Pricing solver         | ✅ | ✅ | ➖ | ✅¹ | ✅¹ | ReWOO |
| ⚖️  Validator             | ✅·0 flags | ✅·0 flags | ➖ | ➖ | ➖ | Critic-Refine |
| 👥 4-Persona Council      | ➖ | ➖ | ➖ | ➖ | ➖ | Critic-Refine (only when validator flags) |
| 📊 Confidence Aggregator  | ✅·0.95 | ✅·0.95 | ✅·0.95 | ✅·0.95 | ✅·0.95 | (pure-Python 8-signal; DEC-0012 adds an LLM rationale paragraph) |
| 🪞 Counterfactual          | ➖ | ➖ | ➖ | ✅·plausible | ✅·plausible | Reflexion (+ ToT inner) |
| ❓ Follow-up Explanation   | ➖ | ➖ | ✅·1 LLM | ➖ | ➖ | Self-Ask + DAG walker |

¹ Inside the Counterfactual fork sub-graph (state-forked re-run).

| Tool | demo-a | demo-b | explain | cf | cf-multi |
|---|---|---|---|---|---|
| `rag_retrieve` (Statutory) | ✅×6 ca_doi | ✅×5 fl_dfs | ➖ | ✅¹ | ✅¹ |
| `fema_nri_risk`            | ✅·NRI-CA-06037 | ✅·NRI-FL-12086 | ➖ | ✅¹ | ✅¹ |
| `flood_zone`               | ✅·Zone X | ✅·Zone AE | ➖ | ✅¹ | ✅¹ |
| `usgs_seismic`             | ✅·PGA 0.93g | ➖ | ➖ | ✅¹ | ✅¹ |
| `noaa_hurricane`           | ➖ | ✅·28 landfalls | ➖ | ➖ | ➖ |
| `ca_fire_zone`             | ✅·Moderate FHSZ | ➖ | ➖ | ✅¹ | ✅¹ |
| `base_premium`             | ✅·CA 2026 | ✅·FL 2026 | ➖ | ✅¹ | ✅¹ |
| `home_value_scaling_factor`| ✅·2.17× | ✅·1.36× | ➖ | ✅¹ | ✅¹ |
| `pricing_multiplier_lookup`| ✅×N | ✅×N | ➖ | ✅×N | ✅×N |
| `cohort_benchmark`         | ✅·Newsweek | ➖ | ➖ | ✅¹ | ✅¹ |
| `citizens_benchmark`       | ➖ | ✅·III FL | ➖ | ➖ | ➖ |
| `coverage_rules`           | ✅ | ✅ | ➖ | ✅¹ | ✅¹ |
| `replacement_cost`         | ✅·$1,005,360 | ✅·$674,016 | ➖ | ✅¹ | ✅¹ |
| `lender_floor`             | ✅ | ✅ | ➖ | ✅¹ | ✅¹ |
| `cea_earthquake_recommender`| ✅·20% ded | ➖ | ➖ | ✅¹ | ✅¹ |
| `fl_hurricane_deductible`  | ➖ | ✅·3 options | ➖ | ➖ | ➖ |
| `wind_mitigation_discount` | ➖ | ✅·advisory | ➖ | ➖ | ➖ |
| `state_diff`               | ➖ | ➖ | ➖ | ✅·−10.7% | ✅·−10.7%³ |
| `rag_retrieve` (Followup)  | ➖ | ➖ | ⚠️² | ➖ | ➖ |

² Available, but the Follow-up agent's keyword router didn't trigger any corpus for the demo question.<br>
³ Same delta as cf (single-axis). The credit_score=600 mutation propagates through Intent and the Counterfactual fork, but the Statutory gate fires on the fork too — CA-PROP103-CREDIT drops credit before the Pricing chain ever sees it. This is the gate working as designed, surfaced honestly as a teaching moment in `walkthrough.html ▸ 09`.

---

## 📈 Run vital signs (real, captured)

| Demo | Total tokens | Total cost | Wall time | Premium range |
|---|---|---|---|---|
| `demo-a` | ~7 K | ~$0.025 | ~22 s | **$4,790 – $7,984** (CA $900K w/ pool, 1 claim) |
| `demo-b` | ~8 K | ~$0.025 | ~25 s | **$5,850 – $9,750** (FL $450K, null credit) |
| `followup-explain` | ~1 K (single Self-Ask call) | ~$0.005 | ~3 s | unchanged from base run |
| `followup-cf` | ~5 K (re-runs Risk + Coverage + Pricing on fork) | ~$0.015 | ~12 s | **delta -$513 / -$855 (-10.7%)** |
| `followup-cf-multi` | ~5 K (same + ToT inner) | ~$0.018 | ~13 s | same delta as `cf` (credit mutation nullified by Prop 103) |

(See each doc for fully-itemised per-agent / per-tool numbers.)
