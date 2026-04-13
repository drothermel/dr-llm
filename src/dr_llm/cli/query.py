from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import typer
from pydantic import ValidationError

from dr_llm.llm.providers.effort import EffortSpec
from dr_llm.llm.providers.reasoning import parse_reasoning_spec
from dr_llm.llm.providers.registry import build_default_registry
from dr_llm.llm.request import parse_llm_request
from dr_llm.logging.events import generation_log_context
from dr_llm.logging.sinks import emit_generation_event

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
    temperature: float | None = typer.Option(
        None,
        help="Sampling temperature (unsupported for headless providers and kimi-code).",
    ),
    top_p: float | None = typer.Option(
        None,
        help="Nucleus sampling parameter (unsupported for headless providers and kimi-code).",
    ),
    max_tokens: int | None = typer.Option(
        None,
        help="Maximum output tokens (unsupported for headless providers; required for anthropic and kimi-code).",
    ),
    effort: EffortSpec = typer.Option(EffortSpec.NA),
    reasoning_json: str | None = typer.Option(
        None,
        help='JSON reasoning config (e.g. {"kind":"budget","tokens":1024}).',
    ),
    metadata_json: str | None = typer.Option(None, help="JSON object metadata."),
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
            parse_reasoning_spec(reasoning_payload)
            if isinstance(reasoning_payload, dict)
            else None
        )
    except ValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc
    messages_payload = common._load_messages(messages_file, message or [])

    request_payload: dict[str, object] = {
        "provider": provider,
        "model": model,
        "messages": messages_payload,
        "effort": effort,
        "reasoning": reasoning,
        "metadata": metadata,
    }
    if temperature is not None:
        request_payload["temperature"] = temperature
    if top_p is not None:
        request_payload["top_p"] = top_p
    if max_tokens is not None:
        request_payload["max_tokens"] = max_tokens

    try:
        request = parse_llm_request(request_payload)
    except ValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc

    registry = build_default_registry()
    try:
        model_provider = registry.get(provider)
        call_id = uuid4().hex

        log_context = {
            "call_id": call_id,
            "provider": request.provider,
            "model": request.model,
            "mode": model_provider.mode,
        }
        with generation_log_context(log_context):
            emit_generation_event(
                event_type="llm_call.started",
                stage="query.before_provider",
                payload={
                    "request": request.model_dump(
                        mode="json",
                        exclude_none=True,
                        exclude_computed_fields=True,
                    )
                },
            )
            try:
                response = model_provider.generate(request)
            except Exception as exc:  # noqa: BLE001
                emit_generation_event(
                    event_type="llm_call.failed",
                    stage="query.provider_exception",
                    payload={
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
                raise

            emit_generation_event(
                event_type="llm_call.succeeded",
                stage="query.after_provider",
                payload={
                    "response": response.model_dump(
                        mode="json",
                        exclude_none=True,
                        exclude_computed_fields=True,
                    )
                },
            )
        common._emit(response.model_dump(mode="json", exclude_computed_fields=True))
    finally:
        registry.close()
