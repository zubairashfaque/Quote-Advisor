"""Application configuration via pydantic-settings + LangSmith bootstrap.

Reads environment variables (and ``.env`` if present), exposes a typed
``Settings`` object, and bootstraps LangSmith / LangChain auto-tracing if
``LANGSMITH_TRACING`` is enabled.

Bootstrap is a side-effect of importing this module's ``get_settings`` in the
CLI entry point, so any LLM call thereafter is auto-traced. No agent code
needs to know about LangSmith.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Typed settings sourced from env + .env."""

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Provider keys (consumed by langchain.init_chat_model on demand)
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")

    # ── LangSmith observability (env-only wire-up, no code changes elsewhere)
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="refocusai", alias="LANGSMITH_PROJECT")
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        alias="LANGSMITH_ENDPOINT",
    )

    # ── Storage paths
    chromadb_dir: Path = Field(default=REPO_ROOT / ".chromadb", alias="QA_CHROMADB_DIR")
    checkpoint_sqlite: Path = Field(
        default=REPO_ROOT / ".langgraph" / "checkpoints.sqlite",
        alias="QA_CHECKPOINT_SQLITE",
    )
    data_dir: Path = Field(default=REPO_ROOT / "data", alias="QA_DATA_DIR")

    # ── Convenience
    repo_root: Path = REPO_ROOT


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton settings accessor.

    Side effects (idempotent):
      1. Copies provider keys (``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY``) from
         ``.env`` -> ``os.environ`` so the OpenAI / Anthropic SDKs pick them up
         (pydantic-settings reads ``.env`` into the Settings object, but does
         not export to ``os.environ`` on its own).
      2. Bootstraps LangSmith env vars if ``LANGSMITH_TRACING`` is true.
      3. Ensures storage directories exist.
    """
    s = Settings()
    _bootstrap_provider_keys(s)
    _bootstrap_langsmith(s)
    _bootstrap_paths(s)
    return s


def _bootstrap_provider_keys(s: Settings) -> None:
    """Bridge provider keys from .env -> os.environ so SDK clients see them."""
    if s.openai_api_key:
        os.environ.setdefault("OPENAI_API_KEY", s.openai_api_key)
    if s.anthropic_api_key:
        os.environ.setdefault("ANTHROPIC_API_KEY", s.anthropic_api_key)


def _bootstrap_langsmith(s: Settings) -> None:
    """Set the env vars LangChain reads at instantiation time.

    LangChain auto-traces every LLM/Tool/Chain when ``LANGSMITH_TRACING`` (or
    the legacy ``LANGCHAIN_TRACING_V2``) is set in the environment before the
    chat model is constructed.

    Resilience: if ``LANGSMITH_TRACING=true`` but the API key is missing or
    rejected, we skip tracing entirely (instead of letting LangChain spam the
    chain with 401-flavoured callback failures that cascade into ReAct loop
    aborts). The user can leave ``LANGSMITH_TRACING=true`` in ``.env`` even
    with an invalid key - this code degrades gracefully.
    """
    import sys

    if not s.langsmith_tracing:
        return

    if not s.langsmith_api_key:
        print(
            "[langsmith] LANGSMITH_TRACING=true but LANGSMITH_API_KEY is empty; "
            "tracing auto-disabled (no callback noise will be emitted).",
            file=sys.stderr,
        )
        return

    if not _probe_langsmith_key(s.langsmith_api_key, s.langsmith_endpoint):
        print(
            f"[langsmith] LANGSMITH_API_KEY rejected by {s.langsmith_endpoint} "
            "(401 Unauthorized); tracing auto-disabled. "
            "Generate a fresh personal token at https://smith.langchain.com/settings "
            "to re-enable.",
            file=sys.stderr,
        )
        return

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", s.langsmith_project)
    os.environ.setdefault("LANGCHAIN_PROJECT", s.langsmith_project)
    os.environ.setdefault("LANGSMITH_ENDPOINT", s.langsmith_endpoint)
    os.environ.setdefault("LANGCHAIN_ENDPOINT", s.langsmith_endpoint)
    os.environ.setdefault("LANGSMITH_API_KEY", s.langsmith_api_key)
    os.environ.setdefault("LANGCHAIN_API_KEY", s.langsmith_api_key)


def _probe_langsmith_key(api_key: str, endpoint: str) -> bool:
    """One-shot ~3-second probe to confirm a LangSmith key is accepted.

    Hits ``GET <endpoint>/sessions?limit=1`` (an auth-required endpoint) with
    the key in ``x-api-key``. Returns True on 2xx, False on 401/403, True on
    any other failure mode (transient network blip, DNS, server error - we
    assume the key is fine and let runtime decide).

    Why probe instead of just trusting the user's key: LangChain's tracing
    layer is wired up at chat-model instantiation; once enabled, every LLM
    call writes traces, and a rejected key produces non-fatal but noisy
    background failures that have been observed to cascade into ReAct loop
    aborts on some langchain versions. Probing once at startup is cheap.

    The ``/info`` endpoint deliberately is NOT used here - it doesn't require
    auth, so it would return 200 regardless of key validity.
    """
    try:
        import urllib.error
        import urllib.request

        url = endpoint.rstrip("/") + "/sessions?limit=1"
        req = urllib.request.Request(
            url, headers={"x-api-key": api_key, "Accept": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as e:
        return False if e.code in (401, 403) else True
    except Exception:
        return True


def _bootstrap_paths(s: Settings) -> None:
    """Ensure storage directories exist."""
    s.chromadb_dir.mkdir(parents=True, exist_ok=True)
    s.checkpoint_sqlite.parent.mkdir(parents=True, exist_ok=True)


def langsmith_run_url_hint(thread_id: str | None) -> str | None:
    """Best-effort URL pointing the reviewer at this run's trace.

    LangSmith's run-id is only known after invocation; this returns the
    project-level URL filterable by thread_id.
    """
    s = get_settings()
    if not s.langsmith_tracing:
        return None
    project = s.langsmith_project
    base = "https://smith.langchain.com"
    if thread_id:
        return f"{base}/projects/{project}/traces?metadata=thread_id:{thread_id}"
    return f"{base}/projects/{project}/traces"
