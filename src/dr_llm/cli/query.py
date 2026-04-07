from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

import typer
from pydantic import ValidationError

from dr_llm.logging import emit_generation_event, generation_log_context
from dr_llm.pool.db.repository import PoolDb
from dr_llm.llm.providers.effort import EffortSpec
from dr_llm.llm.providers.registry import build_default_registry
from dr_llm.llm.request import LlmRequest
from dr_llm.llm.providers.reasoning import parse_reasoning_spec

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
    effort: EffortSpec = typer.Option(EffortSpec.NA),
    reasoning_json: str | None = typer.Option(
        None,
        help='JSON reasoning config (e.g. {"kind":"budget","tokens":1024}).',
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
            parse_reasoning_spec(reasoning_payload)
            if isinstance(reasoning_payload, dict)
            else None
        )
    except ValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc
    messages_payload = common._load_messages(messages_file, message or [])

    try:
        request = LlmRequest(
            provider=provider,
            model=model,
            messages=messages_payload,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            effort=effort,
            reasoning=reasoning,
            metadata=metadata,
        )
    except ValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc

    registry = build_default_registry()
    try:
        model_provider = registry.get(provider)
        call_id = external_call_id or uuid4().hex

        repository: PoolDb | None = None
        if record:
            repository = common._repo(dsn, min_pool_size, max_pool_size)

        log_context = {
            "call_id": call_id,
            "run_id": run_id,
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
                if repository is not None:
                    try:
                        repository.record_call(
                            request=request,
                            response=None,
                            run_id=run_id,
                            status="failed",
                            mode=model_provider.mode,
                            error_text=str(exc),
                            external_call_id=external_call_id,
                            metadata=metadata,
                            call_id=call_id,
                        )
                    except Exception as rec_exc:  # noqa: BLE001
                        logging.getLogger(__name__).warning(
                            "Failed to record failed call (call_id=%s, run_id=%s): %s",
                            call_id,
                            run_id,
                            rec_exc,
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
            if repository is not None:
                try:
                    repository.record_call(
                        request=request,
                        response=response,
                        run_id=run_id,
                        status="success",
                        mode=model_provider.mode,
                        external_call_id=external_call_id,
                        metadata=metadata,
                        call_id=call_id,
                    )
                except Exception as rec_exc:  # noqa: BLE001
                    logging.getLogger(__name__).warning(
                        "Failed to record successful call (call_id=%s, run_id=%s): %s",
                        call_id,
                        run_id,
                        rec_exc,
                    )

        common._emit(response.model_dump(mode="json", exclude_computed_fields=True))
    finally:
        if repository is not None:
            repository.close()
        registry.close()
