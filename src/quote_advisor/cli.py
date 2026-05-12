"""Multi-Agent Quote Advisor CLI.

Usage:
    python -m quote_advisor.cli --profile examples/profile_a.json [--thread-id demo-a]
    python -m quote_advisor.cli --thread-id demo-a --followup "Why is this quote expensive?"

Flags:
    --profile         path to customer profile JSON
    --followup        single follow-up question (within the same thread_id)
    --thread-id       thread identifier for MemorySaver checkpointing
    --verbose         emit per-node DecisionTrace + LangSmith URL to stderr
    --seed            log a seed value (Anthropic/OpenAI do not honour seed for chat)
    --no-rag          disable RAG retrieval at the Follow-up agent
    --llm-trace       print the resolved provider:model for every role at startup
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .configuration import get_settings, langsmith_run_url_hint
from .decision_trace import render_summary, reset_counter
from .graph import build_graph
from .guardrails.input_validation import InputValidationError, validate_input
from .llm_registry import all_resolved
from .schemas import QuoteOutput

app = typer.Typer(
    add_completion=False,
    help="Multi-Agent Quote Advisor CLI.",
    no_args_is_help=True,
)
console = Console(stderr=True)


def _print_llm_trace() -> None:
    table = Table(title="Resolved per-agent LLM assignments", show_lines=False)
    table.add_column("Role", style="cyan", no_wrap=True)
    table.add_column("Model", style="green")
    for role, model in all_resolved().items():
        table.add_row(role, model)
    console.print(table)


def _load_profile(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise typer.BadParameter(f"profile file not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


@app.command()
def run(
    profile: str | None = typer.Option(None, "--profile", help="Path to customer profile JSON"),
    followup: str | None = typer.Option(None, "--followup", help="Follow-up question (same thread_id)"),
    thread_id: str = typer.Option("demo-default", "--thread-id", help="Thread ID for MemorySaver"),
    verbose: bool = typer.Option(False, "--verbose", help="Print DecisionTrace + LangSmith URL to stderr"),
    seed: int | None = typer.Option(None, "--seed", help="Logged seed value (informational only)"),
    no_rag: bool = typer.Option(False, "--no-rag", help="Disable RAG retrieval at Follow-up"),
    llm_trace: bool = typer.Option(False, "--llm-trace", help="Print resolved per-agent models at startup"),
) -> None:
    """Run a single Quote Advisor invocation."""
    if not profile and not followup:
        raise typer.BadParameter("Provide --profile (new quote) or --followup (existing thread).")

    settings = get_settings()  # bootstraps LangSmith if enabled
    if llm_trace:
        _print_llm_trace()

    if seed is not None:
        console.print(f"[dim]seed={seed} (informational; Anthropic/OpenAI chat models do not honour client-side seeds).[/dim]")

    raw_profile = _load_profile(profile) if profile else {}

    if raw_profile:
        try:
            validate_input(raw_profile, decision_trace=[])
        except InputValidationError as exc:
            console.print(f"[red]Profile validation failed:[/red] {exc}")
            sys.exit(1)

    reset_counter()
    graph = build_graph()

    invocation: dict = {
        "raw_profile": raw_profile,
        "thread_id":   thread_id,
        "followup_question": followup,
        "decision_trace": [],
        "warnings": [],
    }
    if no_rag:
        invocation["disable_rag"] = True

    config = {"configurable": {"thread_id": thread_id}}

    final_state = graph.invoke(invocation, config=config)

    output = final_state.get("output")
    if isinstance(output, dict):
        output = QuoteOutput.model_validate(output)
    if output is None:
        console.print("[red]No output produced.[/red]")
        sys.exit(2)

    # ---- Compose the structured QuoteOutput JSON to stdout --------------------
    spec_shape = {
        "risk_factors":           [rf.model_dump() if hasattr(rf, "model_dump") else rf for rf in output.risk_factors],
        "recommended_coverages":  [c.model_dump() if hasattr(c, "model_dump") else c for c in output.recommended_coverages],
        "premium_range":          output.premium_range.model_dump(),
        "explanation":            output.explanation,
        "confidence_score":       output.confidence_score,
        "warnings":               output.warnings,
    }
    sys.stdout.write(json.dumps(spec_shape, indent=2, default=str))
    sys.stdout.write("\n")
    sys.stdout.flush()

    if verbose:
        console.print("\n[bold]DecisionTrace[/bold]")
        console.print(render_summary(output.decision_trace))

        # DEC-0011 — surface the StatutoryAgent's ReAct trajectory + fallback flag
        statutory_audits = [
            n for n in (output.decision_trace or []) if getattr(n, "agent", "") == "StatutoryAgent"
        ]
        if statutory_audits:
            audit = statutory_audits[0]
            payload = getattr(audit, "payload", {}) or {}
            phase = payload.get("phase", "agent")
            traj = payload.get("react_trajectory", []) or []
            console.print(f"\n[bold]Statutory agent[/bold] · phase=[cyan]{phase}[/cyan] · "
                          f"retrievals={len(traj)} · rules={len(payload.get('rule_ids', []))}")
            if phase == "fallback":
                console.print(f"  [yellow][FALLBACK] reason: {payload.get('reason', 'unknown')}[/yellow]")
            for i, step in enumerate(traj, 1):
                console.print(
                    f"  {i}. corpus=[cyan]{step.get('corpus')}[/cyan] "
                    f"jurisdiction={step.get('jurisdiction')} "
                    f"q='{step.get('query', '')[:60]}' "
                    f"→ {step.get('n_chunks', 0)} chunks "
                    f"({', '.join(eid for eid in (step.get('evidence_ids') or []) if eid)})"
                )

        console.print(f"\n[bold]Confidence breakdown:[/bold] {output.confidence_breakdown}")

        # DEC-0012 — pull out the rationale paragraph if the explainer ran
        cb = output.confidence_breakdown
        if cb is not None and getattr(cb, "rationale_summary", None):
            console.print(f"\n[bold]Confidence rationale (LLM):[/bold]\n  {cb.rationale_summary}")

        url = langsmith_run_url_hint(thread_id)
        if url:
            console.print(f"\n[bold]LangSmith trace:[/bold] {url}")
        if settings.langsmith_tracing:
            console.print("[dim]Tip: filter LangSmith by metadata.thread_id == "
                          f"'{thread_id}' to scope to this run.[/dim]")


if __name__ == "__main__":
    app()
