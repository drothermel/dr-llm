"""Console formatting helpers for dr-llm demos."""

from __future__ import annotations

from collections.abc import Iterable

from rich.console import Console
from rich.text import Text

_console = Console()


def header(msg: str) -> None:
    _console.print()
    _console.print(Text(f"=== {msg} ===", style="bold"))
    _console.print()


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


def command(cmd: str) -> None:
    _console.print(Text(f"$ {cmd}", style="bold"))


def command_hint(label: str, cmd: str) -> None:
    _console.print(Text.assemble(f"  {label}:  ", (cmd, "cyan")))


def print_list(
    header: str,
    values: Iterable[str],
    *,
    empty: str = "none",
    use_step: bool = True,
) -> None:
    if use_step:
        step(header)
    else:
        _console.print(header)

    items = list(values)
    if not items:
        _console.print(f"  {empty}")
        return

    for value in items:
        _console.print(f"  - {value}")


__all__ = [
    "command",
    "command_hint",
    "fail",
    "header",
    "ok",
    "print_list",
    "step",
    "warn",
]
