"""Async wrappers that delegate sync backends via asyncio.to_thread."""

from __future__ import annotations

import asyncio
from collections.abc import Callable


async def run_in_thread[T](fn: Callable[[], T]) -> T:
    """Run a sync callable on the default thread pool."""
    return await asyncio.to_thread(fn)
