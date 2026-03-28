#!/usr/bin/env python3
"""Demo: show supported providers and currently available providers."""

from __future__ import annotations

import os
import subprocess

import typer

from dr_llm.providers import (
    available_provider_names,
    build_default_registry,
    supported_provider_names,
    supported_provider_statuses,
)

app = typer.Typer()

BOLD = "\033[1m"
CYAN = "\033[0;36m"
YELLOW = "\033[0;33m"
RESET = "\033[0m"
RED = "\033[0;31m"
GREEN = "\033[0;32m"


def step(msg: str) -> None:
    print(f"\n{BOLD}{CYAN}-- {msg}{RESET}\n")


def ok(msg: str) -> None:
    print(f"{GREEN}  ok: {msg}{RESET}")


def fail(msg: str) -> None:
    print(f"{RED}  FAIL: {msg}{RESET}")


def run_cli_streaming(*args: str) -> None:
    cmd = ["uv", "run", "dr-llm", *args]
    print(f"{BOLD}$ {' '.join(cmd)}{RESET}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"command exited with status {result.returncode}")


@app.command()
def main() -> None:
    """Show providers and demo catalog sync/list commands for available ones."""
    if not os.getenv("DR_LLM_DATABASE_URL"):
        fail(
            "DR_LLM_DATABASE_URL is not set. Start a local database first with "
            "'source ./scripts/start-test-postgres.sh', then rerun "
            "'uv run python scripts/demo-providers.py'."
        )
        raise typer.Exit(1)

    registry = build_default_registry()
    statuses = supported_provider_statuses(registry)

    step("1. Supported providers")
    for provider in supported_provider_names(registry):
        print(f"  - {provider}")

    step("2. Available providers")
    available = available_provider_names(registry)
    if available:
        for provider in available:
            print(f"  - {provider}")
    else:
        print("  none")
        raise typer.Exit(1)

    unavailable = [status for status in statuses if not status.available]
    if unavailable:
        step("3. Missing requirements")
        for status in unavailable:
            reasons = [f"missing env {env_var}" for env_var in status.missing_env_vars]
            reasons.extend(
                f"missing executable {executable}"
                for executable in status.missing_executables
            )
            print(f"{YELLOW}  - {status.provider}: {', '.join(reasons)}{RESET}")

    failed_providers: list[str] = []
    for idx, provider in enumerate(available, start=1):
        step(f"4.{idx}. Provider: {provider}")
        try:
            run_cli_streaming("models", "sync", "--provider", provider)
            print()
            run_cli_streaming("models", "list", "--provider", provider)
            ok(f"completed provider demo for {provider}")
        except Exception as exc:
            fail(f"{provider}: {exc}")
            failed_providers.append(provider)

    step("5. Summary")
    succeeded = [provider for provider in available if provider not in failed_providers]
    print(f"  succeeded: {len(succeeded)}")
    if succeeded:
        print(f"  providers: {', '.join(succeeded)}")
    if failed_providers:
        print(f"{YELLOW}  failed: {', '.join(failed_providers)}{RESET}")


if __name__ == "__main__":
    app()
