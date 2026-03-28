from __future__ import annotations

from pathlib import Path

import typer
from pydantic import ValidationError

from dr_llm.client import LlmClient
from dr_llm.generation.models import LlmRequest, ReasoningConfig
from dr_llm.providers import build_default_registry
from dr_llm.storage import PostgresRepository

from . import common


def register(app: typer.Typer) -> None:
    app.command("query")(query)


def query(
    provider: str = typer.Option(..., help="Provider key registered in dr-llm."),
    model: str = typer.Option(..., help="Model identifier for the provider."),
    message: list[str] = typer.Option(
        None, "--message", help="User message. Repeatable."
    ),
    messages_file: Path | None = typer.Option(
        None, help="Path to JSON messages payload."
    ),
    temperature: float | None = typer.Option(None),
    top_p: float | None = typer.Option(None),
    max_tokens: int | None = typer.Option(None),
    reasoning_json: str | None = typer.Option(
        None,
        help='JSON reasoning config (e.g. {"effort":"high"} or {"max_tokens":2000}).',
    ),
    metadata_json: str | None = typer.Option(None, help="JSON object metadata."),
    run_id: str | None = typer.Option(None),
    external_call_id: str | None = typer.Option(None),
    record: bool = typer.Option(True, "--record/--no-record"),
    dsn: str | None = typer.Option(None, envvar="DR_LLM_DATABASE_URL"),
    min_pool_size: int = typer.Option(4),
    max_pool_size: int = typer.Option(64),
) -> None:
    """Execute a single LLM query through the unified provider interface."""
    metadata = (
        common._parse_json(metadata_json, arg_name="metadata_json", expected=dict) or {}
    )
    reasoning_payload = common._parse_json(
        reasoning_json, arg_name="reasoning_json", expected=dict
    )
    try:
        reasoning = (
            ReasoningConfig(**reasoning_payload)
            if isinstance(reasoning_payload, dict)
            else None
        )
    except ValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc
    messages_payload = common._load_messages(messages_file, message or [])

    repository: PostgresRepository | None = None
    try:
        if record:
            repository = common._repo(dsn, min_pool_size, max_pool_size)
        client = LlmClient(registry=build_default_registry(), repository=repository)
        try:
            request = LlmRequest(
                provider=provider,
                model=model,
                messages=messages_payload,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                reasoning=reasoning,
                metadata=metadata,
            )
        except ValidationError as exc:
            raise typer.BadParameter(str(exc)) from exc
        response = client.query(
            request,
            run_id=run_id,
            external_call_id=external_call_id,
            metadata=metadata,
        )
        common._emit(response.model_dump(mode="json", exclude_computed_fields=True))
    finally:
        if repository is not None:
            repository.close()
