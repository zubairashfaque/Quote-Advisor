.PHONY: help sync ingest demo-a demo-b followup-explain followup-cf followup-cf-multi clean

PY ?= poetry run python

help:
	@echo "Multi-Agent Quote Advisor — make targets"
	@echo "  make sync                — install dependencies via poetry"
	@echo "  make ingest              — build the 6 RAG corpora into ChromaDB"
	@echo "  make demo-a              — Profile A (CA \$$900K, pool, 1 claim, credit 700)"
	@echo "  make demo-b              — Profile B (FL \$$450K, no pool, credit null)"
	@echo "  make followup-explain    — explanation follow-up on Profile A"
	@echo "  make followup-cf         — single-mutation counterfactual on Profile A"
	@echo "  make followup-cf-multi   — multi-axis counterfactual on Profile A"
	@echo "  make clean               — remove .chromadb/, .langgraph/, *.sqlite"

sync:
	poetry install

ingest:
	$(PY) -m quote_advisor.rag.ingest

demo-a:
	$(PY) -m quote_advisor.cli --profile examples/profile_a.json --thread-id demo-a

demo-b:
	$(PY) -m quote_advisor.cli --profile examples/profile_b.json --thread-id demo-b

followup-explain:
	$(PY) -m quote_advisor.cli --thread-id demo-a --followup "Why is this quote expensive?"

followup-cf:
	$(PY) -m quote_advisor.cli --thread-id demo-a --followup "What if I removed the pool?"

followup-cf-multi:
	$(PY) -m quote_advisor.cli --thread-id demo-a --followup "What if I removed the pool and raised the deductible to \$$5,000?"

clean:
	rm -rf .chromadb .langgraph *.sqlite *.sqlite3
