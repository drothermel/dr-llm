"""Console formatting helpers for dr-llm demos."""

from __future__ import annotations

from rich.console import Console
from rich.text import Text

BOLD = "\033[1m"
CYAN = "\033[0;36m"
GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[0;33m"
RESET = "\033[0m"

_console = Console()


def step(msg: str) -> None:
    _console.print()
    _console.print(Text(f"-- {msg}", style="bold cyan"))
    _console.print()


def ok(msg: str) -> None:
    _console.print(Text(f"  ok: {msg}", style="green"))


def fail(msg: str) -> None:
    _console.print(Text(f"  FAIL: {msg}", style="red"))


def warn(msg: str) -> None:
    _console.print(Text(f"  warn: {msg}", style="yellow"))


__all__ = [
    "BOLD",
    "CYAN",
    "GREEN",
    "RED",
    "RESET",
    "YELLOW",
    "fail",
    "ok",
    "step",
    "warn",
]
