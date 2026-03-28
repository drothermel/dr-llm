#!/usr/bin/env python3
"""Demo: show supported providers and currently available providers."""

from __future__ import annotations

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


def step(msg: str) -> None:
    print(f"\n{BOLD}{CYAN}-- {msg}{RESET}\n")


@app.command()
def main() -> None:
    """Show supported providers, available providers, and missing requirements."""
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


if __name__ == "__main__":
    app()
