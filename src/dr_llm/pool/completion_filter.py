"""Completion state filter for pool sample queries."""

from __future__ import annotations

from typing import Literal

type CompletionFilter = Literal["all", "incomplete", "complete", "error"]
