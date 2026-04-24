"""Read-only handle for inspecting an existing pool.

Library-facing primitive that lets consumers open a pool by project + name
and inspect its samples, pending queue, metadata, and progress without
manually wiring :class:`DbRuntime`, :class:`PoolSchema`, and the various
stores. The reader composes a private :class:`PoolStore` and exposes only
its read-side methods.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import psycopg
from pydantic import BaseModel, ConfigDict, computed_field
from sqlalchemy import Column, MetaData, Table, Text, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import ProgrammingError

from dr_llm.pool.db.runtime import DbConfig, DbRuntime
from dr_llm.pool.db.schema import PoolSchema, _VALID_NAME_RE
from dr_llm.pool.errors import PoolNotFoundError, PoolSchemaNotPersistedError
from dr_llm.pool.key_filter import PoolKeyFilter
from dr_llm.pool.pending.pending_sample import PendingSample
from dr_llm.pool.pending.pending_status import PendingStatus, PendingStatusCounts
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.pool_store import SCHEMA_METADATA_KEY, PoolStore
from dr_llm.project.errors import ProjectNotFoundError
from dr_llm.project.project_service import maybe_get_project


class PoolProgress(BaseModel):
    """Snapshot of pool fill progress.

    ``samples_total`` is the count of rows in the pool's samples table
    (i.e. finalized samples). ``pending_counts`` covers the remaining queue
    buckets reported on cards and progress surfaces: ``pending``, ``leased``,
    and ``failed``. Promoted queue rows remain inspectable directly on the
    pending table but are excluded here because they already contribute to
    ``samples_total``. The ``in_flight`` and ``is_complete`` properties are
    exposed as pydantic ``computed_field``s so they appear in
    :meth:`model_dump` output.
    """

    model_config = ConfigDict(frozen=True)

    samples_total: int
    pending_counts: PendingStatusCounts

    @computed_field
    @property
    def in_flight(self) -> int:
        return self.pending_counts.in_flight

    @computed_field
    @property
    def is_complete(self) -> bool:
        return self.pending_counts.in_flight == 0


def _validate_pool_name(pool_name: str) -> None:
    if not _VALID_NAME_RE.match(pool_name):
        raise ValueError(
            f"pool_name must be lowercase alphanumeric with underscores, "
            f"starting with a letter; got {pool_name!r}"
        )


def _load_schema_from_db(runtime: DbRuntime, pool_name: str) -> PoolSchema:
    """Load a persisted :class:`PoolSchema` from a pool's metadata table.

    Builds an ad-hoc :class:`Table` mirroring the fixed shape of pool
    metadata tables (see :meth:`PoolTables._build_metadata_table`) so the
    lookup works without already having a :class:`PoolSchema` in hand.
    Translates ``UndefinedTable`` errors into :class:`PoolNotFoundError`;
    a missing schema row into :class:`PoolSchemaNotPersistedError`.
    """
    _validate_pool_name(pool_name)

    ad_hoc = MetaData()
    table = Table(
        f"pool_{pool_name}_metadata",
        ad_hoc,
        Column("pool_name", Text, nullable=False),
        Column("key", Text, nullable=False),
        Column("value_json", JSONB, nullable=False),
    )
    stmt = select(table.c.value_json).where(
        table.c.pool_name == pool_name,
        table.c.key == SCHEMA_METADATA_KEY,
    )

    try:
        with runtime.connect() as conn:
            value = conn.execute(stmt).scalar_one_or_none()
    except ProgrammingError as exc:
        orig = getattr(exc, "orig", None)
        if isinstance(orig, psycopg.errors.UndefinedTable):
            raise PoolNotFoundError(
                f"No metadata table found for pool {pool_name!r}; "
                f"the pool has not been created."
            ) from exc
        raise

    if value is None:
        raise PoolSchemaNotPersistedError(
            f"Pool {pool_name!r} exists but has no persisted schema in its "
            f"metadata table (key {SCHEMA_METADATA_KEY!r}). Call "
            f"PoolReader.from_runtime(runtime, schema=...) with the schema "
            f"explicitly, or re-run PoolStore.ensure_schema() to backfill it."
        )
    return PoolSchema(**value)


class PoolReader:
    """Read-only handle to an existing pool.

    Library-facing primitive: consumers open a pool by project + name and
    inspect samples / pending / metadata / progress without manually
    wiring :class:`DbRuntime` + :class:`PoolSchema` + :class:`PoolStore`.
    The reader composes a private :class:`PoolStore` and exposes only its
    read-side methods; write methods (insert, acquire, promote) are not
    accessible.

    Owns its :class:`DbRuntime` when constructed via :meth:`open`; borrows
    it when constructed via :meth:`from_runtime`. :meth:`close` and
    context-manager exit dispose the owned runtime only.

    Not thread-safe; use one reader per thread. Iterators returned by
    :meth:`samples` and :meth:`pending` hold a database connection for
    their lifetime — fully consume them or call :meth:`close`. The
    eager :meth:`samples_list` / :meth:`pending_list` variants are safer
    for callers that don't need streaming.
    """

    schema: PoolSchema
    pool_name: str

    def __init__(
        self,
        *,
        schema: PoolSchema,
        runtime: DbRuntime,
        owns_runtime: bool,
    ) -> None:
        self.schema = schema
        self.pool_name = schema.name
        self._runtime = runtime
        self._owns_runtime = owns_runtime
        self._store = PoolStore(schema, runtime)
        self._closed = False

    @classmethod
    def open(cls, project_name: str, pool_name: str) -> PoolReader:
        """Open a pool by project name and pool name.

        Resolves the project DSN via :func:`maybe_get_project`, constructs
        a fresh :class:`DbRuntime`, and loads the persisted
        :class:`PoolSchema` from the pool's metadata table. The reader
        owns the runtime and disposes it on :meth:`close`.

        Raises:
            ProjectNotFoundError: project does not exist or has no DSN.
            PoolNotFoundError: pool's metadata table does not exist.
            PoolSchemaNotPersistedError: pool exists but its schema row is
                missing (pool was created before this feature shipped).
        """
        project = maybe_get_project(project_name)
        if project is None:
            raise ProjectNotFoundError(f"Project {project_name!r} not found")
        if project.dsn is None:
            raise ProjectNotFoundError(
                f"Project {project_name!r} has no DSN; is the container running?"
            )
        runtime = DbRuntime(DbConfig(dsn=project.dsn))
        try:
            schema = _load_schema_from_db(runtime, pool_name)
        except Exception:
            runtime.close()
            raise
        return cls(schema=schema, runtime=runtime, owns_runtime=True)

    @classmethod
    def from_runtime(cls, runtime: DbRuntime, *, schema: PoolSchema) -> PoolReader:
        """Construct a reader from an existing runtime and explicit schema.

        Use this when you already have a :class:`DbRuntime` to reuse, when
        the pool's schema is not persisted (e.g. created before this
        feature shipped), or when you want a tightly-controlled lifetime
        for testing. The returned reader does NOT take ownership of the
        runtime — :meth:`close` is a no-op for the runtime.
        """
        return cls(schema=schema, runtime=runtime, owns_runtime=False)

    def samples(
        self,
        *,
        key_filter: PoolKeyFilter | None = None,
    ) -> Iterator[PoolSample]:
        """Stream samples; see :meth:`PoolStore.iter_samples` for filter semantics.

        Holds a connection for the iterator's lifetime — fully consume it
        or prefer :meth:`samples_list`.
        """
        return self._store.iter_samples(key_filter=key_filter)

    def samples_list(
        self,
        *,
        key_filter: PoolKeyFilter | None = None,
    ) -> list[PoolSample]:
        """Eagerly materialize samples into a list."""
        return self._store.bulk_load(key_filter=key_filter)

    def pending(
        self,
        *,
        key_filter: PoolKeyFilter | None = None,
        status: PendingStatus | Iterable[PendingStatus] | None = None,
    ) -> Iterator[PendingSample]:
        """Stream pending samples; see :meth:`PendingStore.iter_pending` for filter semantics.

        When ``status`` is ``None``, only in-flight statuses (``pending``
        and ``leased``) are returned, matching the worker-facing default.
        Pass an explicit set to inspect ``promoted`` or ``failed`` rows.
        """
        return self._store.pending.iter_pending(
            key_filter=key_filter,
            status=status,
        )

    def pending_list(
        self,
        *,
        key_filter: PoolKeyFilter | None = None,
        status: PendingStatus | Iterable[PendingStatus] | None = None,
    ) -> list[PendingSample]:
        """Eagerly materialize pending samples into a list."""
        return self._store.pending.bulk_load(
            key_filter=key_filter,
            status=status,
        )

    def progress(self) -> PoolProgress:
        """Return a snapshot of pool fill progress.

        Issues two queries: ``COUNT(*)`` on the samples table plus the
        existing :meth:`PendingStore.status_counts` aggregate.
        """
        return PoolProgress(
            samples_total=self._store.sample_count(),
            pending_counts=self._store.pending.status_counts(),
        )

    def metadata_get(self, key: str) -> dict[str, Any] | None:
        """Read a single metadata value by key."""
        return self._store.metadata.get(key)

    def metadata_prefix(self, prefix: str) -> dict[str, dict[str, Any]]:
        """Read all metadata entries whose key starts with ``prefix``.

        Useful for scanning consumer-owned axis metadata like
        ``prompt_template/`` or ``data_sample/`` without writing raw SQL.
        Pass ``""`` to load every metadata entry in the pool.
        """
        return dict(self._store.metadata.iter_prefix(prefix))

    def close(self) -> None:
        """Dispose the owned :class:`DbRuntime`, if any. Idempotent."""
        if self._closed:
            return
        if self._owns_runtime:
            self._runtime.close()
        self._closed = True

    def __enter__(self) -> PoolReader:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
