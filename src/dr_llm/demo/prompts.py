"""Shared prompts for demo scripts."""

from __future__ import annotations

from enum import StrEnum


class DemoPrompts(StrEnum):
    TWO_PLUS_TWO = "What is 2+2? Answer in one sentence."
    EXACT_OK = "Reply with exactly OK."
    PROGRAMMING_HAIKU = "Write a haiku about programming."


__all__ = ["DemoPrompts"]
