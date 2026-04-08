from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from enum import StrEnum
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

TWorkItem = TypeVar("TWorkItem")
TResult = TypeVar("TResult")
TBackendState = TypeVar("TBackendState", bound=BaseModel)


class ErrorDecision(StrEnum):
    retry = "retry"
    fail = "fail"


ErrorAction = ErrorDecision


type ProcessFn[TWorkItem, TResult] = Callable[[TWorkItem], TResult]


class WorkerBackend(Protocol[TWorkItem, TResult, TBackendState]):
    def claim(self, *, worker_id: str, lease_seconds: int) -> list[TWorkItem]: ...

    def complete(self, *, item: TWorkItem, result: TResult, worker_id: str) -> None: ...

    def handle_process_error(
        self,
        *,
        item: TWorkItem,
        worker_id: str,
        exc: Exception,
    ) -> ErrorDecision: ...

    def snapshot(self) -> TBackendState | None: ...

    def process_context(
        self,
        *,
        item: TWorkItem,
        worker_id: str,
    ) -> AbstractContextManager[Any]: ...
