# Troubleshooting

Common operational issues and their fixes. Referenced from [README Section 15](../README.md#15-troubleshooting).

## "openai.AuthenticationError: Incorrect API key"

Set `OPENAI_API_KEY` in `.env`. Verify with `python -c "from quote_advisor.configuration import get_settings; print(bool(get_settings().openai_api_key))"`.

## "anthropic.AuthenticationError: invalid x-api-key"

You've flipped at least one role to an `anthropic:*` model. Set `ANTHROPIC_API_KEY` in `.env`, or revert the override.

## Fetcher reports `[SYNTH]` for `fema_nri` or `fema_nfhl`

The live FEMA endpoints (`hazards.fema.gov`) failed -- usually a TLS-handshake termination caused by a network-side TLS interceptor (corporate proxy / antivirus / firewall). The fetcher has automatically written a curated synthetic snapshot anchored on real publicly-disclosed FEMA values (LA County overall_score 99.94, Miami panel 12086C0312L AE BFE 8 ft, etc.). Demos run normally. To diagnose: `curl -v https://hazards.fema.gov/`. If curl also fails, the issue is upstream of the Python process; the synthetic data is operative until the network clears.

## "RuntimeError: flood_zone: cache empty"

Run `python data/scripts/fetch_real_data.py --only fema_nfhl` to refresh the FEMA NFHL cache.

## `make ingest` fails with "model download timed out"

The `bge-small-en-v1.5` embedding model downloads once from HuggingFace on first ingest. Re-run `make ingest`; `sentence-transformers` resumes the download.

## CLI hangs at "Resolved per-agent LLM assignments"

LangSmith bootstrap is waiting on a network call. Set `LANGSMITH_TRACING=false` in `.env` if you don't have network access.

## Counterfactual returns `"plausibility_status": "refused"`

The trial delta exceeded +/-50%. The reflection notes are in `counterfactual.reflexion_notes`; the agent retried once. Persistence across turns is on by default -- the next follow-up turn will see the prior reflections.

## Stale checkpoint after schema change

Run `make clean` to wipe `.langgraph/checkpoints.sqlite` and `.chromadb/`. Existing thread state will be lost; demos use fresh `thread_id`s anyway.

## LangSmith URL never appears

`--verbose` is required. Also confirm `LANGSMITH_TRACING=true` in `.env` and that you ran `get_settings()` (the CLI does this implicitly).

## Out-of-state profile (e.g. Texas)

The `STATE-SUPPORTED` rule routes any non-CA / non-FL profile to `informational` mode; confidence is capped. Add the state to `data/tables/statutory_rules.json` if you need to extend coverage.
