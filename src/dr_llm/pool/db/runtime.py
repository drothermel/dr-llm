from __future__ import annotations

import threading
from collections.abc import Generator
from contextlib import contextmanager
from os import getenv
from time import sleep

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import create_engine
from sqlalchemy.engine import Connection, make_url
from sqlalchemy.pool import QueuePool

from dr_llm.errors import TransientPersistenceError


_LEGACY_TABLES = (
    "artifacts",
    "llm_call_responses",
    "llm_call_requests",
    "llm_calls",
    "run_parameters",
    "runs",
    "tool_call_dead_letters",
    "tool_results",
    "tool_calls",
    "session_events",
    "session_turns",
    "sessions",
)


class DbConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    dsn: str = Field(
        default_factory=lambda: getenv(
            "DR_LLM_DATABASE_URL", "postgresql://localhost/dr_llm"
        )
    )
    min_pool_size: int = 4
    max_pool_size: int = 64
    statement_timeout_ms: int | None = None
    application_name: str = "dr_llm"
    open_on_init: bool = False
    pool_open_retries: int = 3
    pool_open_retry_backoff_seconds: float = 0.1


class DbRuntime:
    def __init__(self, config: DbConfig) -> None:
        self.config = config
        self.engine = create_engine(
            _sqlalchemy_dsn(self.config.dsn),
            future=True,
            poolclass=QueuePool,
            pool_size=self.config.min_pool_size,
            max_overflow=max(0, self.config.max_pool_size - self.config.min_pool_size),
            pool_pre_ping=True,
            connect_args={"application_name": self.config.application_name},
        )
        self._legacy_cleanup_lock = threading.Lock()
        self._legacy_cleanup_done = False
        self._engine_lock = threading.Lock()
        self._engine_opened = False
        if self.config.open_on_init:
            self.open_pool()

    def close(self) -> None:
        with self._engine_lock:
            self.engine.dispose()
            self._engine_opened = False

    def open_pool(self) -> None:
        if self._engine_opened:
            return
        with self._engine_lock:
            if self._engine_opened:
                return
            retries = max(1, int(self.config.pool_open_retries))
            last_exc: Exception | None = None
            for attempt in range(1, retries + 1):
                try:
                    with self.engine.connect():
                        self._engine_opened = True
                        return
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if attempt >= retries:
                        break
                    sleep(max(0.0, float(self.config.pool_open_retry_backoff_seconds)))
            raise TransientPersistenceError(
                f"Failed to open connection pool after {retries} attempts: {last_exc}"
            ) from last_exc

    def initialize(
        self,
        *,
        allow_destructive_cleanup: bool = False,
        dedicated_schema: str | None = None,
    ) -> None:
        self.open_pool()
        self.cleanup_legacy_tables(
            allow_destructive_cleanup=allow_destructive_cleanup,
            dedicated_schema=dedicated_schema,
        )

    @contextmanager
    def connect(self) -> Generator[Connection, None, None]:
        self.open_pool()
        with self.engine.connect() as conn:
            self._configure_connection(conn)
            yield conn

    @contextmanager
    def begin(self) -> Generator[Connection, None, None]:
        self.open_pool()
        with self.engine.begin() as conn:
            self._configure_connection(conn)
            yield conn

    def init_schema(
        self,
        *,
        allow_destructive_cleanup: bool = False,
        dedicated_schema: str | None = None,
    ) -> None:
        self.cleanup_legacy_tables(
            allow_destructive_cleanup=allow_destructive_cleanup,
            dedicated_schema=dedicated_schema,
        )

    def cleanup_legacy_tables(
        self,
        *,
        allow_destructive_cleanup: bool = False,
        dedicated_schema: str | None = None,
    ) -> None:
        if not allow_destructive_cleanup:
            return
        validated_schema = _validate_dedicated_schema(dedicated_schema)
        if self._legacy_cleanup_done:
            return
        with self._legacy_cleanup_lock:
            if self._legacy_cleanup_done:
                return
            with self.begin() as conn:
                current_schema = conn.exec_driver_sql(
                    "SELECT current_schema()"
                ).scalar_one()
                if current_schema != validated_schema:
                    raise ValueError(
                        "Destructive legacy cleanup requires the current schema to "
                        f"match dedicated_schema={validated_schema!r}; "
                        f"got {current_schema!r}"
                    )
                for table_name in _LEGACY_TABLES:
                    conn.exec_driver_sql(
                        f'DROP TABLE IF EXISTS "{validated_schema}"."{table_name}" CASCADE'
                    )
            self._legacy_cleanup_done = True

    def _configure_connection(self, conn: Connection) -> None:
        if self.config.statement_timeout_ms is None:
            return
        conn.exec_driver_sql(
            "SET statement_timeout = %s",
            (int(self.config.statement_timeout_ms),),
        )


def _sqlalchemy_dsn(dsn: str) -> str:
    url = make_url(dsn)
    if "+" in url.drivername:
        return url.render_as_string(hide_password=False)
    return url.set(drivername=f"{url.drivername}+psycopg").render_as_string(
        hide_password=False
    )


def _validate_dedicated_schema(dedicated_schema: str | None) -> str:
    if dedicated_schema is None or not dedicated_schema.strip():
        raise ValueError(
            "Destructive legacy cleanup requires a non-empty dedicated_schema"
        )
    return dedicated_schema.strip()
