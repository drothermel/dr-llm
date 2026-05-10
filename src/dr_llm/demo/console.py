"""Console formatting helpers for dr-llm demos."""

from __future__ import annotations

BOLD = "\033[1m"
CYAN = "\033[0;36m"
GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[0;33m"
RESET = "\033[0m"


def step(msg: str) -> None:
    print(f"\n{BOLD}{CYAN}-- {msg}{RESET}\n")


def ok(msg: str) -> None:
    print(f"{GREEN}  ok: {msg}{RESET}")


def fail(msg: str) -> None:
    print(f"{RED}  FAIL: {msg}{RESET}")


def warn(msg: str) -> None:
    print(f"{YELLOW}  warn: {msg}{RESET}")


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
