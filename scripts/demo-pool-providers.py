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
import os
import shutil
import subprocess
from typing import Any

import typer
from pydantic import BaseModel

from dr_llm.pool.models import PoolSample
from dr_llm.pool.schema import KeyColumn, PoolSchema
from dr_llm.pool.store import PoolStore
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


class ProviderSpec(BaseModel):
    name: str
    env_var: str | None = None
    cli_tool: str | None = None
    default_model: str


PROVIDERS: list[ProviderSpec] = [
    # API providers — detected by env var
    ProviderSpec(name="openai", env_var="OPENAI_API_KEY", default_model="gpt-4o-mini"),
    ProviderSpec(
        name="anthropic",
        env_var="ANTHROPIC_API_KEY",
        default_model="claude-sonnet-4-20250514",
    ),
    ProviderSpec(
        name="google", env_var="GOOGLE_API_KEY", default_model="gemini-2.0-flash"
    ),
    ProviderSpec(
        name="openrouter",
        env_var="OPENROUTER_API_KEY",
        default_model="openai/gpt-4o-mini",
    ),
    ProviderSpec(name="glm", env_var="ZAI_API_KEY", default_model="glm-4-flash"),
    ProviderSpec(
        name="minimax",
        env_var="MINIMAX_API_KEY",
        default_model="MiniMax-M2.5-highspeed",
    ),
    # Headless providers — detected by CLI tool presence (own OAuth/auth)
    ProviderSpec(name="claude-code", cli_tool="claude", default_model="sonnet"),
    ProviderSpec(name="codex", cli_tool="codex", default_model="gpt-5.4-mini"),
    # Headless variants — need CLI tool AND API key for third-party routing
    ProviderSpec(
        name="claude-code-minimax",
        cli_tool="claude",
        env_var="MINIMAX_API_KEY",
        default_model="MiniMax-M2.5-highspeed",
    ),
    ProviderSpec(
        name="claude-code-kimi",
        cli_tool="claude",
        env_var="KIMI_API_KEY",
        default_model="kimi-for-coding",
    ),
]

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


def _has_cli_tool(tool: str) -> bool:
    return shutil.which(tool) is not None


def detect_providers() -> list[ProviderSpec]:
    """Detect which providers are available based on env vars and CLI tools."""
    available: list[ProviderSpec] = []
    for spec in PROVIDERS:
        has_env = spec.env_var is None or os.getenv(spec.env_var)
        has_tool = spec.cli_tool is None or _has_cli_tool(spec.cli_tool)
        if has_env and has_tool:
            available.append(spec)
        else:
            reasons: list[str] = []
            if spec.env_var and not os.getenv(spec.env_var):
                reasons.append(f"{spec.env_var} not set")
            if spec.cli_tool and not _has_cli_tool(spec.cli_tool):
                reasons.append(f"'{spec.cli_tool}' CLI not found")
            warn(f"{spec.name}: {', '.join(reasons)}")
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


def resolve_model(project: str, spec: ProviderSpec) -> str:
    """Sync catalog, list models, pick default or first available."""
    run_cli("--project", project, "models", "sync", "--provider", spec.name)
    result = run_cli(
        "--project", project, "models", "list", "--provider", spec.name, "--json"
    )
    models = result.get("models", [])
    if not models:
        raise RuntimeError(f"No models found for {spec.name}")

    model_ids = [m["model"] for m in models]
    if spec.default_model in model_ids:
        return spec.default_model
    print(f"  default model '{spec.default_model}' not found, using '{model_ids[0]}'")
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
    available = detect_providers()
    if not available:
        print(
            f"\n{RED}No providers available. Set API keys or install CLI tools.{RESET}"
        )
        raise typer.Exit(1)
    print(
        f"\n  Found {len(available)} providers: {', '.join(s.name for s in available)}"
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

        for i, spec in enumerate(available, 1):
            step(f"4.{i}. Provider: {spec.name}")
            try:
                # Resolve model
                model = resolve_model(project_name, spec)
                ok(f"Using model: {model}")

                # Show model info
                info = show_model(project_name, spec.name, model)
                display = info.get("display_name", model)
                ctx = info.get("context_window")
                ctx_str = f", context={ctx}" if ctx else ""
                ok(f"Model info: {display}{ctx_str}")

                # Query
                is_headless = spec.cli_tool is not None
                print(f"  Querying {spec.name}/{model}...")
                response = query_provider(
                    project_name, spec.name, model, prompt, is_headless=is_headless
                )
                text = (response.get("text") or "")[:80].replace("\n", " ")
                latency = response.get("latency_ms", "?")
                ok(f"Response ({latency}ms): {text}")

                # Store in pool
                store_result(store, spec.name, model, prompt, response)
                ok("Stored in pool")
                succeeded.append(spec.name)

            except Exception as exc:
                fail(f"{spec.name}: {exc}")
                failed_providers.append(spec.name)

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
