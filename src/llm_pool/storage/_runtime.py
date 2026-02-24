from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from hashlib import sha256
from os import getenv
from pathlib import Path
from time import sleep
from typing import Any, Generator, LiteralString, cast

import psycopg
from psycopg import errors, sql
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, ConfigDict, Field

from llm_pool.errors import TransientPersistenceError


_SCHEMA_PATH = Path(__file__).with_name("schema_bootstrap_pg.sql")
_MIGRATION_PATHS = [
    Path(__file__).with_name("schema_migration_20260224_llm_call_response_columns.sql"),
]


class StorageConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    dsn: str = Field(
        default_factory=lambda: getenv(
            "LLM_POOL_DATABASE_URL", "postgresql://localhost/llm_pool"
        )
    )
    min_pool_size: int = 4
    max_pool_size: int = 64
    statement_timeout_ms: int | None = None
    application_name: str = "llm_pool"
    open_on_init: bool = False
    pool_open_retries: int = 3
    pool_open_retry_backoff_seconds: float = 0.1


def is_retryable_db_error(exc: BaseException) -> bool:
    if isinstance(
        exc,
        (
            psycopg.OperationalError,
            psycopg.InterfaceError,
            errors.DeadlockDetected,
            errors.SerializationFailure,
        ),
    ):
        return True
    if isinstance(exc, TransientPersistenceError):
        return True
    return False


def hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


class StorageRuntime:
    def __init__(self, config: StorageConfig) -> None:
        self.config = config
        self.pool = ConnectionPool(
            self.config.dsn,
            min_size=self.config.min_pool_size,
            max_size=self.config.max_pool_size,
            open=False,
        )
        self.schema_lock = threading.Lock()
        self.schema_initialized = False
        self._pool_lock = threading.Lock()
        self._pool_opened = False
        if self.config.open_on_init:
            self.open_pool()

    def close(self) -> None:
        with self._pool_lock:
            self.pool.close()
            self._pool_opened = False

    def open_pool(self) -> None:
        if self._pool_opened:
            return
        with self._pool_lock:
            if self._pool_opened:
                return
            retries = max(1, int(self.config.pool_open_retries))
            last_exc: Exception | None = None
            for attempt in range(1, retries + 1):
                try:
                    self.pool.open(wait=True)
                    self._pool_opened = True
                    return
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if attempt >= retries:
                        break
                    sleep(max(0.0, float(self.config.pool_open_retry_backoff_seconds)))
            raise TransientPersistenceError(
                f"Failed to open connection pool after {retries} attempts: {last_exc}"
            ) from last_exc

    def initialize(self) -> None:
        self.open_pool()
        self.init_schema()

    @contextmanager
    def conn(self) -> Generator[psycopg.Connection[tuple[Any, ...]], None, None]:
        self.open_pool()
        with self.pool.connection() as conn:
            if self.config.statement_timeout_ms is not None:
                conn.execute(
                    "SET statement_timeout = %s",
                    [int(self.config.statement_timeout_ms)],
                )
            yield conn

    def init_schema(self) -> None:
        if self.schema_initialized:
            return
        with self.schema_lock:
            if self.schema_initialized:
                return
            schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")
            with self.conn() as conn:
                conn.execute(sql.SQL(cast(LiteralString, schema_sql)))
                for migration_path in _MIGRATION_PATHS:
                    if not migration_path.exists():
                        continue
                    migration_sql = migration_path.read_text(encoding="utf-8")
                    conn.execute(sql.SQL(cast(LiteralString, migration_sql)))
                conn.commit()
            self.schema_initialized = True
