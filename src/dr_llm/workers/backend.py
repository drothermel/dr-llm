from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from enum import StrEnum
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

TWorkItem = TypeVar("TWorkItem")
TResult = TypeVar("TResult")


class ErrorAction(StrEnum):
    retry = "retry"
    fail = "fail"


type ProcessFn[TWorkItem, TResult] = Callable[[TWorkItem], TResult]
type ProcessContextFactory[TWorkItem] = Callable[
    [TWorkItem, str], AbstractContextManager[Any]
]


class WorkerBackend(Protocol[TWorkItem, TResult]):
    def claim(
        self, *, worker_id: str, limit: int, lease_seconds: int
    ) -> list[TWorkItem]: ...

    def complete(self, *, item: TWorkItem, result: TResult, worker_id: str) -> None: ...

    def handle_process_error(
        self,
        *,
        item: TWorkItem,
        worker_id: str,
        exc: Exception,
        max_retries: int,
    ) -> ErrorAction: ...

    def snapshot(self) -> BaseModel | None: ...
