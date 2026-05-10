#!/usr/bin/env python3
"""Demo: query all available LLM providers and store results in a typed pool.

Demonstrates the hybrid CLI + Python API workflow (Flow 2):
- CLI (subprocess): project create/destroy, models sync/list/show, query
- Python API: PoolSchema, PoolStore, PoolSample, bulk_load

Prerequisites:
  1. Docker running (used to spin up a Postgres container for the pool)
  2. At least one provider available:
     - API key env var (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.), or
     - CLI tool installed (claude, codex)

Usage:
  uv run python scripts/demo-pool-providers.py

  The script will:
  - Create a Docker-based Postgres project called 'demo_pool'
  - Detect which LLM providers are available
  - Query each provider and store results in a typed pool
  - Print a summary table of all results
  - Leave the project running so you can inspect the data

  To clean up afterwards:
    uv run dr-llm project destroy demo_pool --yes-really-delete-everything
"""

from __future__ import annotations

from typing import Any

import typer

from dr_llm.demo import (
    DEMO_QUERY_DEFAULT_MODELS,
    DemoPrompts,
    command_hint,
    create_demo_project,
    ensure_docker_available,
    fail,
    list_models_json,
    ok,
    query_json,
    require_demo_project_dsn,
    show_model_json,
    step,
    sync_models_json,
    warn,
)
from dr_llm.llm import (
    EffortSpec,
    ProviderAvailabilityStatus,
    ProviderName,
    build_default_registry,
    default_effort,
    default_reasoning,
)
from dr_llm.pool import (
    DbConfig,
    DbRuntime,
    KeyColumn,
    PoolSample,
    PoolSchema,
    PoolStore,
    PoolSummaryColumn,
    print_pool_summary,
)
from dr_llm.project import ProjectInfo, destroy_project

app = typer.Typer()

DEFAULT_PROMPT = DemoPrompts.TWO_PLUS_TWO
DEFAULT_PROJECT = "demo_pool"
API_TIMEOUT = 120
HEADLESS_TIMEOUT = 300
ANTHROPIC_MAX_TOKENS = 256


POOL_SCHEMA = PoolSchema(
    name="provider_queries",
    key_columns=[KeyColumn(name="provider"), KeyColumn(name="model")],
)


# --- Detection ---


def detect_providers(
    statuses: list[ProviderAvailabilityStatus],
) -> list[ProviderAvailabilityStatus]:
    """Return available providers and print skip reasons for unavailable ones."""
    available: list[ProviderAvailabilityStatus] = []
    for status in statuses:
        if status.available:
            available.append(status)
            continue
        reasons = [f"{env_var} not set" for env_var in status.missing_env_vars]
        reasons.extend(
            f"'{executable}' CLI not found"
            for executable in status.missing_executables
        )
        warn(f"{status.provider}: {', '.join(reasons)}")
    return available


# --- Model Resolution ---


def resolve_model(provider: ProviderName) -> str:
    """Sync catalog, list models, pick default or first available."""
    sync_models_json(provider)
    models = list_models_json(provider)
    if not models:
        raise RuntimeError(f"No models found for {provider}")

    model_ids = [m["model"] for m in models]
    default_model = DEMO_QUERY_DEFAULT_MODELS.get(provider)
    if default_model and default_model in model_ids:
        return default_model
    if default_model:
        print(
            f"  default model '{default_model}' not found, using '{model_ids[0]}'"
        )
    else:
        print(
            f"  no default model configured for {provider}, using '{model_ids[0]}'"
        )
    return model_ids[0]


# --- Query ---


def provider_extra_args(provider: str, model: str) -> list[str]:
    args: list[str] = []
    reasoning = default_reasoning(provider=provider, model=model)
    if reasoning is not None:
        args.extend(
            [
                "--reasoning-json",
                reasoning.model_dump_json(exclude_none=True),
            ]
        )
    effort = default_effort(provider=provider, model=model)
    if effort != EffortSpec.NA:
        args.extend(["--effort", effort.value])
    return args


def query_provider(
    provider: str, model: str, prompt: str, *, is_headless: bool = False
) -> dict[str, Any]:
    """Query a provider via CLI, returning the response dict."""
    timeout = HEADLESS_TIMEOUT if is_headless else API_TIMEOUT
    provider_name = ProviderName(provider)
    extra_args: list[str] = []
    if provider_name in {
        ProviderName.ANTHROPIC,
        ProviderName.KIMI_CODE,
        ProviderName.MINIMAX,
    }:
        extra_args.extend(["--max-tokens", str(ANTHROPIC_MAX_TOKENS)])
    extra_args.extend(provider_extra_args(provider, model))
    return query_json(
        provider,
        model,
        prompt,
        timeout=timeout,
        extra_args=extra_args,
    )


# --- Pool ---


def store_result(
    store: PoolStore,
    provider: str,
    model: str,
    prompt: str,
    response: dict[str, Any],
) -> None:
    store.insert_sample(
        PoolSample(
            key_values={"provider": provider, "model": model},
            request={"prompt": prompt},
            response=response,
            finish_reason=response.get("finish_reason"),
            metadata={
                "cost": response.get("cost") or {},
                "usage": response.get("usage") or {},
                "latency_ms": response.get("latency_ms", 0),
            },
        )
    )


def print_summary(store: PoolStore) -> None:
    """Print a summary table of all pool samples."""
    print_pool_summary(
        store,
        extra_columns=[
            PoolSummaryColumn(
                header="Latency",
                value=latency_cell,
                justify="right",
            ),
            PoolSummaryColumn(
                header="Tokens",
                value=token_cell,
                justify="right",
            ),
        ],
        response_max_chars=50,
    )


def latency_cell(sample: PoolSample) -> str:
    latency = sample.metadata.get("latency_ms", 0)
    return f"{latency}ms"


def token_cell(sample: PoolSample) -> int:
    usage = sample.metadata.get("usage", {})
    if not isinstance(usage, dict):
        return 0
    return int(usage.get("total_tokens", 0) or 0)


# --- Main ---


@app.command()
def main(
    project_name: str = typer.Option(
        DEFAULT_PROJECT, help="Project name for the demo"
    ),
    prompt: str = typer.Option(
        DEFAULT_PROMPT, help="Prompt to send to each provider"
    ),
) -> None:
    """Query all available LLM providers and store results in a typed pool."""
    ensure_docker_available(
        reason="This demo creates a Postgres container to store pool data.",
        recovery_hint="Install Docker, start the daemon, then retry.",
    )

    step("1. Detecting available providers")
    registry = build_default_registry()
    available = detect_providers(registry.availability_statuses())
    if not available:
        fail("No providers available. Set API keys or install CLI tools.")
        raise typer.Exit(1)
    print(
        f"\n  Found {len(available)} providers: "
        f"{', '.join(status.provider for status in available)}"
    )

    step("2. Creating demo project")
    demo_succeeded = False
    runtime: DbRuntime | None = None
    project: ProjectInfo | None = None
    try:
        project = create_demo_project(project_name, replace_existing=True)
        dsn = require_demo_project_dsn(project)
        ok(f"Project '{project_name}' created at {dsn}")

        step("3. Initializing pool")
        runtime = DbRuntime(
            DbConfig(dsn=dsn, min_pool_size=1, max_pool_size=4)
        )
        store = PoolStore(POOL_SCHEMA, runtime)
        store.ensure_schema()
        ok(f"Pool '{POOL_SCHEMA.name}' ready")

        succeeded: list[str] = []
        failed_providers: list[str] = []

        for i, status in enumerate(available, 1):
            provider = status.provider
            provider_name = ProviderName(provider)
            step(f"4.{i}. Provider: {provider}")
            try:
                # Resolve model
                model = resolve_model(provider_name)
                ok(f"Using model: {model}")

                # Show model info
                info = show_model_json(provider, model)
                display = info.get("display_name", model)
                ctx = info.get("context_window")
                ctx_str = f", context={ctx}" if ctx else ""
                ok(f"Model info: {display}{ctx_str}")

                # Query
                is_headless = registry.get(provider).mode == "headless"
                print(f"  Querying {provider}/{model}...")
                response = query_provider(
                    provider, model, prompt, is_headless=is_headless
                )
                text = (response.get("text") or "")[:80].replace("\n", " ")
                latency = response.get("latency_ms", "?")
                ok(f"Response ({latency}ms): {text}")

                # Store in pool
                store_result(store, provider, model, prompt, response)
                ok("Stored in pool")
                succeeded.append(provider)

            except Exception as exc:
                fail(f"{provider}: {exc}")
                failed_providers.append(provider)

        step("5. Results")
        print_summary(store)

        if failed_providers:
            warn(f"Failed providers: {', '.join(failed_providers)}")

        demo_succeeded = True

    finally:
        if runtime:
            runtime.close()
        if not demo_succeeded and project is not None:
            step("Cleaning up after failure...")
            try:
                destroy_project(project_name)
            except Exception:
                pass

    ok("Demo complete!")
    print(f"Project '{project_name}' is still running with your data.\n")
    command_hint(
        "Stop (preserve data)",
        f"uv run dr-llm project stop {project_name}",
    )
    command_hint(
        "Destroy permanently",
        f"uv run dr-llm project destroy {project_name} --yes-really-delete-everything",
    )


if __name__ == "__main__":
    app()
