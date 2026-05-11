"""Thin subprocess helpers for demo scripts that exercise the dr-llm CLI."""

from __future__ import annotations

import json
import subprocess
from typing import Any

from dr_llm.demo.console import command

DEFAULT_CLI_TIMEOUT = 120


def run_dr_llm_json(
    *args: str,
    timeout: int = DEFAULT_CLI_TIMEOUT,
) -> dict[str, Any]:
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


def run_dr_llm_streaming(*args: str) -> None:
    """Run a dr-llm CLI command, streaming stdout for demo visibility."""
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


def sync_models_json(provider: str) -> dict[str, Any]:
    """Sync a provider model catalog and return verbose JSON output."""
    return run_dr_llm_json(
        "models",
        "sync",
        "--provider",
        provider,
        "--verbose",
    )


def list_models_json(provider: str) -> list[dict[str, Any]]:
    """List provider models from the catalog as JSON records."""
    result = run_dr_llm_json(
        "models",
        "list",
        "--provider",
        provider,
        "--json",
    )
    models = result.get("models", [])
    if not isinstance(models, list):
        raise RuntimeError(f"Expected models list for provider {provider!r}.")
    return models


def show_model_json(provider: str, model: str) -> dict[str, Any]:
    """Show one provider model as a JSON object."""
    return run_dr_llm_json(
        "models",
        "show",
        "--provider",
        provider,
        "--model",
        model,
    )


def query_json(
    provider: str,
    model: str,
    prompt: str,
    *,
    timeout: int,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Query a provider through the CLI, returning the JSON response."""
    args = [
        "query",
        "--provider",
        provider,
        "--model",
        model,
        "--message",
        prompt,
    ]
    if extra_args is not None:
        args.extend(extra_args)
    return run_dr_llm_json(*args, timeout=timeout)


def stream_models_sync(provider: str) -> None:
    """Stream a model catalog sync command for demo output."""
    run_dr_llm_streaming("models", "sync", "--provider", provider)


def stream_models_list(provider: str) -> None:
    """Stream a model catalog list command for demo output."""
    run_dr_llm_streaming("models", "list", "--provider", provider)


__all__ = [
    "DEFAULT_CLI_TIMEOUT",
    "list_models_json",
    "query_json",
    "run_dr_llm_json",
    "run_dr_llm_streaming",
    "show_model_json",
    "stream_models_list",
    "stream_models_sync",
    "sync_models_json",
]
