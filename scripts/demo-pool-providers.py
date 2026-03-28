#!/usr/bin/env python3
"""Demo: query all available LLM providers and store results in a typed pool.

Demonstrates the hybrid CLI + Python API workflow:
- CLI (subprocess): project create/destroy, models sync/list/show, query
- Python API: PoolSchema, PoolStore, PoolSample, bulk_load

Prerequisites:
  - Docker running
  - At least one of: API key env var set, or claude/codex CLI installed
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

import typer

from dr_llm.pool.models import PoolSample
from dr_llm.pool.schema import KeyColumn, PoolSchema
from dr_llm.pool.store import PoolStore
from dr_llm.providers import build_default_registry, supported_provider_statuses
from dr_llm.providers.avail import ProviderAvailabilityStatus
from dr_llm.project.docker import destroy_project
from dr_llm.storage._runtime import StorageConfig, StorageRuntime

app = typer.Typer()

BOLD = "\033[1m"
CYAN = "\033[0;36m"
GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[0;33m"
RESET = "\033[0m"

DEFAULT_PROMPT = "What is 2+2? Answer in one sentence."
DEFAULT_PROJECT = "demo-pool"
API_TIMEOUT = 120
HEADLESS_TIMEOUT = 300


DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-20250514",
    "google": "gemini-2.5-flash",
    "openrouter": "openai/gpt-4o-mini",
    "glm": "glm-4.5",
    "minimax": "MiniMax-M2",
    "claude-code": "sonnet",
    "codex": "gpt-5.4-mini",
    "claude-code-minimax": "MiniMax-M2",
    "claude-code-kimi": "kimi-for-coding",
}

POOL_SCHEMA = PoolSchema(
    name="provider_queries",
    key_columns=[KeyColumn(name="provider"), KeyColumn(name="model")],
)


# --- Helpers ---


def step(msg: str) -> None:
    print(f"\n{BOLD}{CYAN}-- {msg}{RESET}\n")


def ok(msg: str) -> None:
    print(f"{GREEN}  ok: {msg}{RESET}")


def fail(msg: str) -> None:
    print(f"{RED}  FAIL: {msg}{RESET}")


def warn(msg: str) -> None:
    print(f"{YELLOW}  skip: {msg}{RESET}")


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


def run_cli_quiet(*args: str, timeout: int = API_TIMEOUT) -> None:
    """Run a dr-llm CLI command, ignoring output and errors."""
    cmd = ["uv", "run", "dr-llm", *args]
    subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


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


def create_demo_project(project_name: str) -> str:
    """Create a demo project, returning its DSN."""
    run_cli_quiet("project", "destroy", project_name, "--yes-really-delete-everything")
    result = run_cli("project", "create", project_name)
    dsn = result.get("dsn")
    if not dsn:
        raise RuntimeError(f"Project create did not return DSN: {result}")
    return dsn


# --- Model Resolution ---


def resolve_model(project: str, provider: str) -> str:
    """Sync catalog, list models, pick default or first available."""
    run_cli("--project", project, "models", "sync", "--provider", provider, "--verbose")
    result = run_cli(
        "--project", project, "models", "list", "--provider", provider, "--json"
    )
    models = result.get("models", [])
    if not models:
        raise RuntimeError(f"No models found for {provider}")

    model_ids = [m["model"] for m in models]
    default_model = DEFAULT_MODELS.get(provider)
    if default_model and default_model in model_ids:
        return default_model
    if default_model:
        print(f"  default model '{default_model}' not found, using '{model_ids[0]}'")
    else:
        print(f"  no default model configured for {provider}, using '{model_ids[0]}'")
    return model_ids[0]


def show_model(project: str, provider: str, model: str) -> dict[str, Any]:
    """Show model details via CLI."""
    return run_cli(
        "--project", project, "models", "show", "--provider", provider, "--model", model
    )


# --- Query ---


def query_provider(
    project: str, provider: str, model: str, prompt: str, *, is_headless: bool = False
) -> dict[str, Any]:
    """Query a provider via CLI, returning the response dict."""
    timeout = HEADLESS_TIMEOUT if is_headless else API_TIMEOUT
    return run_cli(
        "--project",
        project,
        "query",
        "--provider",
        provider,
        "--model",
        model,
        "--message",
        prompt,
        "--no-record",
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
            payload={
                "prompt": prompt,
                "response_text": response.get("text", ""),
                "finish_reason": response.get("finish_reason"),
                "latency_ms": response.get("latency_ms", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            metadata={
                "cost": response.get("cost") or {},
                "usage": usage,
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
        latency = s.payload.get("latency_ms", 0)
        tokens = s.payload.get("total_tokens", 0)
        text = s.payload.get("response_text", "")
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
    project_name: str = typer.Option(DEFAULT_PROJECT, help="Project name for the demo"),
    prompt: str = typer.Option(DEFAULT_PROMPT, help="Prompt to send to each provider"),
) -> None:
    """Query all available LLM providers and store results in a typed pool."""

    step("1. Detecting available providers")
    registry = build_default_registry()
    available = detect_providers(supported_provider_statuses(registry))
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
    runtime: StorageRuntime | None = None
    try:
        dsn = create_demo_project(project_name)
        ok(f"Project '{project_name}' created")

        step("3. Initializing pool")
        runtime = StorageRuntime(
            StorageConfig(dsn=dsn, min_pool_size=1, max_pool_size=4)
        )
        store = PoolStore(POOL_SCHEMA, runtime)
        store.init_schema()
        ok(
            f"Pool '{POOL_SCHEMA.name}' ready "
            f"(tables: {POOL_SCHEMA.samples_table}, {POOL_SCHEMA.claims_table}, ...)"
        )

        succeeded: list[str] = []
        failed_providers: list[str] = []

        for i, status in enumerate(available, 1):
            provider = status.provider
            step(f"4.{i}. Provider: {provider}")
            try:
                # Resolve model
                model = resolve_model(project_name, provider)
                ok(f"Using model: {model}")

                # Show model info
                info = show_model(project_name, provider, model)
                display = info.get("display_name", model)
                ctx = info.get("context_window")
                ctx_str = f", context={ctx}" if ctx else ""
                ok(f"Model info: {display}{ctx_str}")

                # Query
                is_headless = registry.get(provider).mode == "headless"
                print(f"  Querying {provider}/{model}...")
                response = query_provider(
                    project_name, provider, model, prompt, is_headless=is_headless
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
            print(f"\n{YELLOW}Failed providers: {', '.join(failed_providers)}{RESET}")

        demo_succeeded = True

    finally:
        if runtime:
            runtime.close()
        if not demo_succeeded:
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
