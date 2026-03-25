from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import typer
from pydantic import ValidationError

from dr_llm.benchmark import BenchmarkConfig, OperationMix, run_repository_benchmark
from dr_llm.client import LlmClient
from dr_llm.project.cli import project_app
from dr_llm.project.docker import get_project
from dr_llm.providers import build_default_registry
from dr_llm.session import SessionClient, run_tool_worker
from dr_llm.storage import PostgresRepository, StorageConfig
from dr_llm.tools import ToolExecutor, ToolRegistry
from dr_llm.tools.registry import ToolDefinition
from dr_llm.types import (
    LlmRequest,
    Message,
    ModelCatalogQuery,
    ReasoningConfig,
    RunStatus,
    SessionStartInput,
    SessionStepInput,
    ToolPolicy,
)

app = typer.Typer()
run_app = typer.Typer(help="Run lifecycle commands")
session_app = typer.Typer(help="Session lifecycle commands")
tool_app = typer.Typer(help="Tool worker commands")
worker_app = typer.Typer(help="Tool queue worker commands")
replay_app = typer.Typer(help="Replay and audit commands")
models_app = typer.Typer(help="Model catalog commands")

app.add_typer(run_app, name="run")
app.add_typer(session_app, name="session")
app.add_typer(tool_app, name="tool")
app.add_typer(replay_app, name="replay")
app.add_typer(models_app, name="models")
app.add_typer(project_app, name="project")
tool_app.add_typer(worker_app, name="worker")


@app.callback()
def main(
    project: str | None = typer.Option(None, help="Use a named project's database."),
) -> None:
    """dr-llm CLI"""
    if project is not None:
        info = get_project(project)
        if info is None:
            typer.secho(f"Project '{project}' not found", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
        if info.status != "running":
            typer.secho(
                f"Project '{project}' is {info.status} — start it first",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)
        os.environ["DR_LLM_DATABASE_URL"] = info.dsn


def _emit(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _parse_json(
    value: str | None, *, arg_name: str, expected: type | tuple[type, ...] | None = None
) -> Any:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Invalid JSON for {arg_name}: {exc}") from exc
    if expected is not None and not isinstance(parsed, expected):
        expected_names = (
            ", ".join(t.__name__ for t in expected)
            if isinstance(expected, tuple)
            else expected.__name__
        )
        raise typer.BadParameter(f"{arg_name} must decode to {expected_names}")
    return parsed


def _load_messages(
    messages_file: Path | None,
    messages: list[str],
    *,
    require_nonempty: bool = True,
) -> list[Message]:
    result: list[Message] = []
    if messages_file is not None:
        try:
            payload = json.loads(messages_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(f"messages-file is not valid JSON: {exc}") from exc
        if isinstance(payload, dict):
            payload = payload.get("messages")
        if not isinstance(payload, list):
            raise typer.BadParameter(
                "messages-file must be a JSON list or an object with a 'messages' list"
            )
        for item in payload:
            if not isinstance(item, dict):
                raise typer.BadParameter("messages-file entries must be JSON objects")
            try:
                result.append(Message(**item))
            except ValidationError as exc:
                raise typer.BadParameter(
                    f"Invalid message in messages-file: {exc}"
                ) from exc

    result.extend(Message(role="user", content=content) for content in messages)
    if require_nonempty and not result:
        raise typer.BadParameter(
            "At least one message is required (use --message or --messages-file)"
        )
    return result


def _repo(
    dsn: str | None, min_pool_size: int, max_pool_size: int
) -> PostgresRepository:
    if dsn is None:
        cfg = StorageConfig(min_pool_size=min_pool_size, max_pool_size=max_pool_size)
    else:
        cfg = StorageConfig(
            dsn=dsn,
            min_pool_size=min_pool_size,
            max_pool_size=max_pool_size,
        )
    return PostgresRepository(cfg)


def _load_tool_loader(spec: str) -> Any:
    module_name, sep, attr_name = spec.partition(":")
    if not sep:
        raise typer.BadParameter(
            f"Invalid tool loader '{spec}'. Expected module:function, e.g. unitbench.tools:register_tools"
        )
    module = importlib.import_module(module_name)
    target = getattr(module, attr_name, None)
    if target is None:
        raise typer.BadParameter(f"Tool loader target not found: {spec}")
    if not callable(target):
        raise typer.BadParameter(f"Tool loader is not callable: {spec}")
    return target


def _build_tool_registry(tool_loader: list[str]) -> ToolRegistry:
    registry = ToolRegistry()
    for spec in tool_loader:
        loader = _load_tool_loader(spec)
        loaded = loader(registry)
        if loaded is None:
            continue
        if isinstance(loaded, ToolDefinition):
            registry.register(loaded)
            continue
        if isinstance(loaded, list):
            for tool in loaded:
                if not isinstance(tool, ToolDefinition):
                    raise typer.BadParameter(
                        f"Tool loader '{spec}' returned non-ToolDefinition item in list: {tool!r}"
                    )
                registry.register(tool)
            continue
        raise typer.BadParameter(
            f"Tool loader '{spec}' returned unsupported value. Return None, ToolDefinition, or list[ToolDefinition]."
        )
    return registry


@app.command("providers")
def providers() -> None:
    """List known providers and their declared capabilities."""
    client = LlmClient(registry=build_default_registry())
    data = []
    for name in client.known_providers():
        caps = client.provider_capabilities(name)
        data.append({"provider": name, **caps})
    _emit({"providers": data})


@models_app.command("sync")
def models_sync(
    provider: str | None = typer.Option(None, help="Optional provider key."),
    dsn: str | None = typer.Option(None, envvar="DR_LLM_DATABASE_URL"),
    min_pool_size: int = typer.Option(4),
    max_pool_size: int = typer.Option(64),
) -> None:
    """Sync provider model catalog into PostgreSQL."""
    repository = _repo(dsn, min_pool_size, max_pool_size)
    try:
        client = LlmClient(registry=build_default_registry(), repository=repository)
        results = client.sync_models_detailed(provider=provider)
        _emit(
            {
                "results": [
                    result.model_dump(
                        mode="json",
                        exclude_none=True,
                        exclude_computed_fields=True,
                    )
                    for result in results
                ]
            }
        )
    finally:
        repository.close()


@models_app.command("list")
def models_list(
    provider: str | None = typer.Option(None, help="Optional provider filter."),
    supports_reasoning: bool | None = typer.Option(
        None, help="Optional reasoning support filter."
    ),
    model_contains: str | None = typer.Option(None, help="Substring model filter."),
    limit: int = typer.Option(200),
    offset: int = typer.Option(0),
    json_output: bool = typer.Option(
        False,
        "--json/--no-json",
        help="Emit JSON output.",
    ),
    dsn: str | None = typer.Option(None, envvar="DR_LLM_DATABASE_URL"),
    min_pool_size: int = typer.Option(4),
    max_pool_size: int = typer.Option(64),
) -> None:
    """List models from stored catalog."""
    repository = _repo(dsn, min_pool_size, max_pool_size)
    try:
        registry = build_default_registry()
        if provider is not None:
            try:
                provider = registry.get(provider).name
            except KeyError:
                pass
        client = LlmClient(registry=registry, repository=repository)
        items = client.list_models(
            ModelCatalogQuery(
                provider=provider,
                supports_reasoning=supports_reasoning,
                model_contains=model_contains,
                limit=limit,
                offset=offset,
            )
        )
        if json_output:
            _emit(
                {
                    "models": [
                        item.model_dump(
                            mode="json",
                            exclude_none=True,
                            exclude_computed_fields=True,
                        )
                        for item in items
                    ]
                }
            )
        else:
            static_providers: set[str] = set()
            for item in items:
                typer.echo(item.model)
                if item.source_quality == "static":
                    static_providers.add(item.provider)
            for sp in sorted(static_providers):
                docs_url = ""
                for item in items:
                    if item.provider == sp and item.metadata.get("docs_url"):
                        docs_url = item.metadata["docs_url"]
                        break
                msg = f"\nNote: {sp} models are from a static list and may be out of date."
                if docs_url:
                    msg += f"\nSee {docs_url} for the latest models."
                typer.echo(msg, err=True)
    finally:
        repository.close()


@models_app.command("show")
def models_show(
    provider: str = typer.Option(...),
    model: str = typer.Option(...),
    dsn: str | None = typer.Option(None, envvar="DR_LLM_DATABASE_URL"),
    min_pool_size: int = typer.Option(4),
    max_pool_size: int = typer.Option(64),
) -> None:
    """Show one model from stored catalog."""
    repository = _repo(dsn, min_pool_size, max_pool_size)
    try:
        client = LlmClient(registry=build_default_registry(), repository=repository)
        item = client.show_model(provider=provider, model=model)
        if item is None:
            typer.secho(
                f"Model not found for provider={provider!r} model={model!r}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        _emit(
            item.model_dump(
                mode="json", exclude_none=True, exclude_computed_fields=True
            )
        )
    finally:
        repository.close()


@app.command("query")
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
    tools_json: str | None = typer.Option(
        None, help="JSON list of provider tool definitions."
    ),
    tool_policy: ToolPolicy = typer.Option(ToolPolicy.native_preferred),
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
    metadata = _parse_json(metadata_json, arg_name="metadata_json", expected=dict) or {}
    tools = _parse_json(tools_json, arg_name="tools_json", expected=list)
    reasoning_payload = _parse_json(
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
    messages_payload = _load_messages(messages_file, message or [])

    repository: PostgresRepository | None = None
    try:
        if record:
            repository = _repo(dsn, min_pool_size, max_pool_size)
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
                tools=tools,
                tool_policy=tool_policy,
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
        _emit(response.model_dump(mode="json", exclude_computed_fields=True))
    finally:
        if repository is not None:
            repository.close()


@run_app.command("start")
def run_start(
    run_type: str = typer.Option("generic"),
    status: RunStatus = typer.Option(RunStatus.running),
    run_id: str | None = typer.Option(None),
    metadata_json: str | None = typer.Option(None),
    parameters_json: str | None = typer.Option(
        None, help="Optional JSON object of run parameters."
    ),
    dsn: str | None = typer.Option(None, envvar="DR_LLM_DATABASE_URL"),
    min_pool_size: int = typer.Option(4),
    max_pool_size: int = typer.Option(64),
) -> None:
    """Start or upsert a run record."""
    metadata = _parse_json(metadata_json, arg_name="metadata_json", expected=dict) or {}
    parameters = (
        _parse_json(parameters_json, arg_name="parameters_json", expected=dict) or {}
    )

    repository = _repo(dsn, min_pool_size, max_pool_size)
    try:
        persisted_run_id = repository.start_run(
            run_type=run_type,
            status=status,
            metadata=metadata,
            run_id=run_id,
        )
        written = repository.upsert_run_parameters(
            run_id=persisted_run_id, parameters=parameters
        )
        _emit({"run_id": persisted_run_id, "parameters_written": written})
    finally:
        repository.close()


@run_app.command("finish")
def run_finish(
    run_id: str = typer.Option(...),
    status: RunStatus = typer.Option(...),
    metadata_json: str | None = typer.Option(None),
    dsn: str | None = typer.Option(None, envvar="DR_LLM_DATABASE_URL"),
    min_pool_size: int = typer.Option(4),
    max_pool_size: int = typer.Option(64),
) -> None:
    """Finish a run record."""
    metadata = _parse_json(metadata_json, arg_name="metadata_json", expected=dict)
    repository = _repo(dsn, min_pool_size, max_pool_size)
    try:
        repository.finish_run(run_id=run_id, status=status, metadata=metadata)
        _emit({"run_id": run_id, "status": status.value})
    finally:
        repository.close()


@run_app.command("list-calls")
def run_list_calls(
    run_id: str | None = typer.Option(None, help="Filter by run ID."),
    limit: int = typer.Option(100),
    offset: int = typer.Option(0),
    dsn: str | None = typer.Option(None, envvar="DR_LLM_DATABASE_URL"),
    min_pool_size: int = typer.Option(4),
    max_pool_size: int = typer.Option(64),
) -> None:
    """List recorded LLM calls, optionally filtered by run."""
    repository = _repo(dsn, min_pool_size, max_pool_size)
    try:
        calls = repository.list_calls(run_id=run_id, limit=limit, offset=offset)
        _emit(
            {
                "calls": [
                    call.model_dump(
                        mode="json",
                        exclude_none=True,
                        exclude_computed_fields=True,
                    )
                    for call in calls
                ],
                "count": len(calls),
            }
        )
    finally:
        repository.close()


@run_app.command("benchmark")
def run_benchmark(
    workers: int = typer.Option(64, help="Parallel worker threads."),
    total_operations: int = typer.Option(20000, help="Measured operations to execute."),
    warmup_operations: int | None = typer.Option(
        None, help="Warmup operations before measured phase."
    ),
    max_in_flight: int = typer.Option(
        64, help="Maximum in-flight operations across all workers."
    ),
    run_type: str = typer.Option("benchmark"),
    operation_mix_json: str | None = typer.Option(
        None,
        help='JSON operation mix object (e.g. {"record_call":1,"session_roundtrip":1,"read_calls":1}).',
    ),
    artifact_path: Path | None = typer.Option(
        None, help="Path to write benchmark artifact JSON."
    ),
    max_failure_ratio: float = typer.Option(
        0.0, help="Maximum allowed failure ratio (0.0-1.0)."
    ),
    max_error_samples: int = typer.Option(100, help="Max error samples to record."),
    dsn: str | None = typer.Option(None, envvar="DR_LLM_DATABASE_URL"),
    min_pool_size: int = typer.Option(4),
    max_pool_size: int = typer.Option(64),
) -> None:
    """Run a DB-backed mixed read/write concurrency benchmark."""
    operation_mix_payload = _parse_json(
        operation_mix_json, arg_name="operation_mix_json", expected=dict
    )
    try:
        operation_mix = (
            OperationMix(**operation_mix_payload)
            if isinstance(operation_mix_payload, dict)
            else OperationMix()
        )
        config = BenchmarkConfig(
            workers=workers,
            total_operations=total_operations,
            warmup_operations=warmup_operations,
            max_in_flight=max_in_flight,
            run_type=run_type,
            operation_mix=operation_mix,
            artifact_path=str(artifact_path) if artifact_path is not None else None,
            max_failure_ratio=max_failure_ratio,
            max_error_samples=max_error_samples,
        )
    except ValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc

    repository = _repo(dsn, min_pool_size, max_pool_size)
    try:
        report = run_repository_benchmark(repository=repository, config=config)
        _emit(
            {
                "run_id": report.run_id,
                "status": report.status.value,
                "operations_per_second": report.measured.operations_per_second,
                "p50_latency_ms": report.measured.p50_latency_ms,
                "p95_latency_ms": report.measured.p95_latency_ms,
                "failed_operations": report.measured.failed_operations,
                "artifact_path": report.artifact_path,
            }
        )
    finally:
        repository.close()


@session_app.command("start")
def session_start(
    provider: str = typer.Option(...),
    model: str = typer.Option(...),
    message: list[str] = typer.Option(
        None, "--message", help="User message. Repeatable."
    ),
    messages_file: Path | None = typer.Option(None),
    strategy_mode: ToolPolicy = typer.Option(ToolPolicy.native_preferred),
    reasoning_json: str | None = typer.Option(
        None,
        help="JSON reasoning config to persist for this session's model calls.",
    ),
    metadata_json: str | None = typer.Option(None),
    run_id: str | None = typer.Option(None),
    tool_loader: list[str] = typer.Option(
        None, "--tool-loader", help="module:function loader for tools"
    ),
    dsn: str | None = typer.Option(None, envvar="DR_LLM_DATABASE_URL"),
    min_pool_size: int = typer.Option(4),
    max_pool_size: int = typer.Option(64),
) -> None:
    """Create a session and persist initial messages."""
    metadata = _parse_json(metadata_json, arg_name="metadata_json", expected=dict) or {}
    reasoning_payload = _parse_json(
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
    messages_payload = _load_messages(messages_file, message or [])

    repository = _repo(dsn, min_pool_size, max_pool_size)
    try:
        tool_registry = _build_tool_registry(tool_loader or [])
        llm_client = LlmClient(registry=build_default_registry(), repository=repository)
        client = SessionClient(
            llm_client=llm_client, repository=repository, tool_registry=tool_registry
        )
        try:
            input_payload = SessionStartInput(
                provider=provider,
                model=model,
                messages=messages_payload,
                reasoning=reasoning,
                strategy_mode=strategy_mode,
                metadata=metadata,
                run_id=run_id,
            )
        except ValidationError as exc:
            raise typer.BadParameter(str(exc)) from exc
        handle = client.start_session(input_payload)
        _emit(handle.model_dump(mode="json", exclude_computed_fields=True))
    finally:
        repository.close()


@session_app.command("step")
def session_step(
    session_id: str = typer.Option(...),
    message: list[str] = typer.Option(
        None, "--message", help="User message. Repeatable."
    ),
    messages_file: Path | None = typer.Option(None),
    expected_version: int | None = typer.Option(None),
    inline_tool_execution: bool = typer.Option(
        False,
        "--inline-tool-execution/--queue-tool-execution",
        help="Execute tool calls synchronously in-process instead of queueing for workers.",
    ),
    reasoning_json: str | None = typer.Option(
        None,
        help="Optional JSON reasoning config override for this step.",
    ),
    metadata_json: str | None = typer.Option(None),
    tool_loader: list[str] = typer.Option(
        None, "--tool-loader", help="module:function loader for tools"
    ),
    dsn: str | None = typer.Option(None, envvar="DR_LLM_DATABASE_URL"),
    min_pool_size: int = typer.Option(4),
    max_pool_size: int = typer.Option(64),
) -> None:
    """Advance a session by one model/tool step."""
    metadata = _parse_json(metadata_json, arg_name="metadata_json", expected=dict) or {}
    reasoning_payload = _parse_json(
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
    messages_payload = _load_messages(
        messages_file, message or [], require_nonempty=False
    )

    repository = _repo(dsn, min_pool_size, max_pool_size)
    try:
        tool_registry = _build_tool_registry(tool_loader or [])
        llm_client = LlmClient(registry=build_default_registry(), repository=repository)
        client = SessionClient(
            llm_client=llm_client, repository=repository, tool_registry=tool_registry
        )
        try:
            input_payload = SessionStepInput(
                session_id=session_id,
                messages=messages_payload,
                expected_version=expected_version,
                inline_tool_execution=inline_tool_execution,
                reasoning=reasoning,
                metadata=metadata,
            )
        except ValidationError as exc:
            raise typer.BadParameter(str(exc)) from exc
        result = client.step_session(input_payload)
        _emit(result.model_dump(mode="json", exclude_computed_fields=True))
    finally:
        repository.close()


@session_app.command("resume")
def session_resume(
    session_id: str = typer.Option(...),
    dsn: str | None = typer.Option(None, envvar="DR_LLM_DATABASE_URL"),
    min_pool_size: int = typer.Option(4),
    max_pool_size: int = typer.Option(64),
) -> None:
    """Load current materialized session state."""
    repository = _repo(dsn, min_pool_size, max_pool_size)
    try:
        state = repository.get_session(session_id=session_id)
        _emit(state.model_dump(mode="json", exclude_computed_fields=True))
    finally:
        repository.close()


@session_app.command("cancel")
def session_cancel(
    session_id: str = typer.Option(...),
    reason: str = typer.Option(...),
    dsn: str | None = typer.Option(None, envvar="DR_LLM_DATABASE_URL"),
    min_pool_size: int = typer.Option(4),
    max_pool_size: int = typer.Option(64),
) -> None:
    """Cancel a session with a reason."""
    repository = _repo(dsn, min_pool_size, max_pool_size)
    try:
        llm_client = LlmClient(registry=build_default_registry(), repository=repository)
        client = SessionClient(
            llm_client=llm_client, repository=repository, tool_registry=ToolRegistry()
        )
        client.cancel_session(session_id=session_id, reason=reason)
        _emit({"session_id": session_id, "status": "canceled", "reason": reason})
    finally:
        repository.close()


@worker_app.command("run")
def tool_worker_run(
    tool_loader: list[str] = typer.Option(
        ..., "--tool-loader", help="module:function loader for tools"
    ),
    worker_id: str | None = typer.Option(None),
    lease_seconds: int = typer.Option(60),
    batch_size: int = typer.Option(8),
    idle_sleep_seconds: float = typer.Option(0.5),
    max_loops: int | None = typer.Option(None),
    max_attempts_before_dead_letter: int = typer.Option(3),
    dsn: str | None = typer.Option(None, envvar="DR_LLM_DATABASE_URL"),
    min_pool_size: int = typer.Option(4),
    max_pool_size: int = typer.Option(64),
) -> None:
    """Run a DB-backed tool worker loop."""
    repository = _repo(dsn, min_pool_size, max_pool_size)
    try:
        tool_registry = _build_tool_registry(tool_loader)
        executor = ToolExecutor(registry=tool_registry)
        effective_worker_id = worker_id or f"tool-worker-{uuid4().hex[:8]}"
        stats = run_tool_worker(
            repository=repository,
            executor=executor,
            worker_id=effective_worker_id,
            lease_seconds=lease_seconds,
            batch_size=batch_size,
            idle_sleep_seconds=idle_sleep_seconds,
            max_loops=max_loops,
            max_attempts_before_dead_letter=max_attempts_before_dead_letter,
        )
        _emit({"worker_id": effective_worker_id, "stats": stats})
    finally:
        repository.close()


@replay_app.command("session")
def replay_session(
    session_id: str = typer.Option(...),
    include_events: bool = typer.Option(True, "--include-events/--no-events"),
    dsn: str | None = typer.Option(None, envvar="DR_LLM_DATABASE_URL"),
    min_pool_size: int = typer.Option(4),
    max_pool_size: int = typer.Option(64),
) -> None:
    """Replay session state from event log."""
    repository = _repo(dsn, min_pool_size, max_pool_size)
    try:
        state = repository.get_session(session_id=session_id)
        messages = repository.replay_session_messages(session_id=session_id)
        payload: dict[str, Any] = {
            "state": state.model_dump(mode="json", exclude_computed_fields=True),
            "replayed_messages": messages,
            "message_count": len(messages),
        }
        if include_events:
            events = repository.load_session_events(session_id=session_id)
            payload["events"] = [
                event.model_dump(mode="json", exclude_computed_fields=True)
                for event in events
            ]
            payload["event_count"] = len(events)
        _emit(payload)
    finally:
        repository.close()


if __name__ == "__main__":
    app()
