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

    def initialize(self) -> None:
        self.open_pool()

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
