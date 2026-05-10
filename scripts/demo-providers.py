#!/usr/bin/env python3
"""Demo: discover providers and sync/browse model catalogs (Flow 1).

No database or Docker required. Uses file-based catalog cache.

Usage:
  uv run python scripts/demo-providers.py

  The script will:
  - List all supported and available providers
  - For each available provider, sync its model catalog and list models
  - Catalog data is cached locally at ~/.dr_llm/catalog_cache/
"""

from __future__ import annotations

import subprocess

import typer

from dr_llm.demo import command, fail, ok, print_list, step, warn
from dr_llm.llm import build_default_registry

app = typer.Typer()


def run_cli_streaming(*args: str) -> None:
    cmd = ["uv", "run", "dr-llm", *args]
    command(" ".join(cmd))
    try:
        subprocess.run(cmd, check=True, text=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        detail = f": {stderr}" if stderr else ""
        raise RuntimeError(
            f"command {' '.join(exc.cmd)} exited with status {exc.returncode}{detail}"
        ) from exc


@app.command()
def main() -> None:
    """Show providers and demo catalog sync/list commands for available ones."""
    registry = build_default_registry()
    statuses = registry.availability_statuses()

    print_list("1. Supported providers", registry.sorted_names())

    available = registry.available_names(statuses=statuses)
    print_list("2. Available providers", available)

    unavailable = [status for status in statuses if not status.available]
    if unavailable:
        step("3. Missing requirements")
        for status in unavailable:
            reasons = [
                f"missing env {env_var}" for env_var in status.missing_env_vars
            ]
            reasons.extend(
                f"missing executable {executable}"
                for executable in status.missing_executables
            )
            warn(f"{status.provider}: {', '.join(reasons)}")

    if not available:
        raise typer.Exit(1)

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
    succeeded = [
        provider for provider in available if provider not in failed_providers
    ]
    print(f"  succeeded: {len(succeeded)}")
    if succeeded:
        print(f"  providers: {', '.join(succeeded)}")
    if failed_providers:
        warn(f"failed: {', '.join(failed_providers)}")


if __name__ == "__main__":
    app()
