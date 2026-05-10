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

import json
import shutil
import subprocess
from typing import Any

import typer

from _demo_utils import (
    BOLD,
    CYAN,
    GREEN,
    RED,
    RESET,
    YELLOW,
    fail,
    ok,
    step,
    warn,
)
from dr_llm.llm import ProviderAvailabilityStatus, build_default_registry
from dr_llm.pool import (
    DbConfig,
    DbRuntime,
    KeyColumn,
    PoolSample,
    PoolSchema,
    PoolStore,
)
from dr_llm.project import (
    CreateProjectRequest,
    ProjectInfo,
    create_project,
    destroy_project,
    maybe_get_project,
)

app = typer.Typer()

DEFAULT_PROMPT = "What is 2+2? Answer in one sentence."
DEFAULT_PROJECT = "demo_pool"
API_TIMEOUT = 120
HEADLESS_TIMEOUT = 300
ANTHROPIC_MAX_TOKENS = 256


DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-20250514",
    "google": "gemini-2.5-flash",
    "glm": "glm-4.5",
    "minimax": "MiniMax-M2",
    "claude-code": "claude-sonnet-4-6",
    "codex": "gpt-5.4-mini",
    "kimi-code": "kimi-for-coding",
}

# Per-provider extra CLI args needed to satisfy reasoning/effort validation.
# Reasoning and effort requirements vary by provider/model on this branch;
# we send the minimum valid config so the demo can exercise every provider
# end to end without picking reasoning-tuned model variants.
PROVIDER_EXTRA_ARGS: dict[str, list[str]] = {
    "google": [
        "--reasoning-json",
        '{"kind":"google","thinking_level":"budget","budget_tokens":1}',
    ],
    "glm": ["--reasoning-json", '{"kind":"glm","thinking_level":"off"}'],
    "codex": ["--reasoning-json", '{"kind":"codex","thinking_level":"low"}'],
    "openrouter": [
        "--reasoning-json",
        '{"kind":"openrouter","enabled":false}',
    ],
    "minimax": [
        "--reasoning-json",
        '{"kind":"anthropic","thinking_level":"na"}',
        "--effort",
        "low",
    ],
    "kimi-code": [
        "--reasoning-json",
        '{"kind":"anthropic","thinking_level":"adaptive"}',
        "--effort",
        "low",
    ],
    "claude-code": [
        "--reasoning-json",
        '{"kind":"anthropic","thinking_level":"adaptive"}',
        "--effort",
        "low",
    ],
}

POOL_SCHEMA = PoolSchema(
    name="provider_queries",
    key_columns=[KeyColumn(name="provider"), KeyColumn(name="model")],
)


# --- Helpers ---


def run_cli(*args: str, timeout: int = API_TIMEOUT) -> dict[str, Any]:
    """Run a dr-llm CLI command and return parsed JSON output."""
    cmd = ["uv", "run", "dr-llm", *args]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()[:500]
        raise RuntimeError(f"CLI command failed: {' '.join(args)}\n{stderr}")
    return json.loads(result.stdout)


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


# --- Project ---


def create_demo_project(project_name: str) -> ProjectInfo:
    """Create a demo project, destroying any existing one first."""
    existing = maybe_get_project(project_name)
    if existing is not None:
        destroy_project(project_name)
    return create_project(CreateProjectRequest(project_name=project_name))


# --- Model Resolution ---


def resolve_model(provider: str) -> str:
    """Sync catalog, list models, pick default or first available."""
    run_cli("models", "sync", "--provider", provider, "--verbose")
    result = run_cli("models", "list", "--provider", provider, "--json")
    models = result.get("models", [])
    if not models:
        raise RuntimeError(f"No models found for {provider}")

    model_ids = [m["model"] for m in models]
    default_model = DEFAULT_MODELS.get(provider)
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


def show_model(provider: str, model: str) -> dict[str, Any]:
    """Show model details via CLI."""
    return run_cli("models", "show", "--provider", provider, "--model", model)


# --- Query ---


def query_provider(
    provider: str, model: str, prompt: str, *, is_headless: bool = False
) -> dict[str, Any]:
    """Query a provider via CLI, returning the response dict."""
    timeout = HEADLESS_TIMEOUT if is_headless else API_TIMEOUT
    args = [
        "query",
        "--provider",
        provider,
        "--model",
        model,
        "--message",
        prompt,
    ]
    if provider in {"anthropic", "kimi-code", "minimax"}:
        args.extend(["--max-tokens", str(ANTHROPIC_MAX_TOKENS)])
    args.extend(PROVIDER_EXTRA_ARGS.get(provider, []))
    return run_cli(
        *args,
        timeout=timeout,
    )


# --- Pool ---


def store_result(
    store: PoolStore,
    provider: str,
    model: str,
    prompt: str,
    response: dict[str, Any],
) -> None:
    """Insert a query result into the pool."""
    usage = response.get("usage") or {}
    store.insert_sample(
        PoolSample(
            key_values={"provider": provider, "model": model},
            request={"prompt": prompt},
            response=response,
            finish_reason=response.get("finish_reason"),
            metadata={
                "cost": response.get("cost") or {},
                "usage": usage,
                "latency_ms": response.get("latency_ms", 0),
            },
        )
    )


def print_summary(store: PoolStore) -> None:
    """Print a summary table of all pool samples."""
    samples = store.bulk_load()
    if not samples:
        print("\nNo samples in pool.")
        return

    # Column widths
    prov_w = max(len(s.key_values["provider"]) for s in samples)
    prov_w = max(prov_w, len("Provider"))
    model_w = max(len(s.key_values["model"]) for s in samples)
    model_w = max(model_w, len("Model"))
    resp_w = 50

    header = (
        f"{'Provider':<{prov_w}} | {'Model':<{model_w}} | "
        f"{'Latency':>7} | {'Tokens':>6} | Response"
    )
    sep = f"{'-' * prov_w}-+-{'-' * model_w}-+-{'-' * 7}-+-{'-' * 6}-+-{'-' * resp_w}"

    print(f"\n{BOLD}=== Pool Summary: {POOL_SCHEMA.name} ==={RESET}\n")
    print(header)
    print(sep)

    for s in samples:
        provider = s.key_values["provider"]
        model = s.key_values["model"]
        latency = (s.metadata or {}).get("latency_ms", 0)
        usage = (s.metadata or {}).get("usage", {})
        tokens = usage.get("total_tokens", 0)
        text = (s.response or {}).get("text", "")
        text_trunc = text[: resp_w - 3] + "..." if len(text) > resp_w else text
        text_trunc = text_trunc.replace("\n", " ")

        print(
            f"{provider:<{prov_w}} | {model:<{model_w}} | "
            f"{latency:>5}ms | {tokens:>6} | {text_trunc}"
        )

    print(
        f"\nTotal: {len(samples)} samples from "
        f"{len({s.key_values['provider'] for s in samples})} providers"
    )


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
    if not shutil.which("docker"):
        fail(
            "Docker is required but not found.\n"
            "  This demo creates a Postgres container to store pool data.\n"
            "  Install Docker and ensure it's running, then retry."
        )
        raise typer.Exit(1)

    step("1. Detecting available providers")
    registry = build_default_registry()
    available = detect_providers(registry.availability_statuses())
    if not available:
        print(
            f"\n{RED}No providers available. Set API keys or install CLI tools.{RESET}"
        )
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
        project = create_demo_project(project_name)
        assert project.dsn is not None
        ok(f"Project '{project_name}' created at {project.dsn}")

        step("3. Initializing pool")
        runtime = DbRuntime(
            DbConfig(dsn=project.dsn, min_pool_size=1, max_pool_size=4)
        )
        store = PoolStore(POOL_SCHEMA, runtime)
        store.ensure_schema()
        ok(f"Pool '{POOL_SCHEMA.name}' ready")

        succeeded: list[str] = []
        failed_providers: list[str] = []

        for i, status in enumerate(available, 1):
            provider = status.provider
            step(f"4.{i}. Provider: {provider}")
            try:
                # Resolve model
                model = resolve_model(provider)
                ok(f"Using model: {model}")

                # Show model info
                info = show_model(provider, model)
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
            print(
                f"\n{YELLOW}Failed providers: {', '.join(failed_providers)}{RESET}"
            )

        demo_succeeded = True

    finally:
        if runtime:
            runtime.close()
        if not demo_succeeded and project is not None:
            print(f"\n{BOLD}Cleaning up after failure...{RESET}")
            try:
                destroy_project(project_name)
            except Exception:
                pass

    print(f"\n{BOLD}{GREEN}Demo complete!{RESET}")
    print(f"Project '{project_name}' is still running with your data.\n")
    print(
        f"  Stop (preserve data):  {CYAN}uv run dr-llm project stop {project_name}{RESET}"
    )
    print(
        f"  Destroy permanently:   {CYAN}uv run dr-llm project destroy {project_name} "
        f"--yes-really-delete-everything{RESET}"
    )


if __name__ == "__main__":
    app()
