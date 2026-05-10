"""Read-only handle for inspecting an existing pool."""

from __future__ import annotations

from collections.abc import Iterator

from dr_llm.pool.completion_filter import CompletionFilter
from dr_llm.pool.db import DbRuntime, PoolSchema
from dr_llm.pool.db.catalog import load_schema
from dr_llm.pool.db.key_filter import PoolKeyFilter
from dr_llm.pool.errors import PoolNotFoundError
from dr_llm.pool.pool_progress import PoolProgress
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.pool_store import PoolStore


class PoolReader:
    """Read-only handle to an existing pool.

    Composes a private :class:`PoolStore` and exposes only read-side
    methods. Not thread-safe; use one reader per thread.
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
    def open(cls, pool_name: str, *, runtime: DbRuntime) -> PoolReader:
        """Open a pool by name, loading its schema from the catalog table.

        Borrows the provided runtime (does not take ownership).

        Raises:
            PoolNotFoundError: No catalog entry for this pool name.
        """
        schema = load_schema(runtime, pool_name)
        if schema is None:
            raise PoolNotFoundError(
                f"Pool {pool_name!r} not found in the catalog. "
                f"Was ensure_schema() called for this pool?"
            )
        return cls(schema=schema, runtime=runtime, owns_runtime=False)

    @classmethod
    def from_runtime(cls, runtime: DbRuntime, *, schema: PoolSchema) -> PoolReader:
        """Construct a reader from an existing runtime and explicit schema.

        The returned reader does NOT take ownership of the runtime.
        """
        return cls(schema=schema, runtime=runtime, owns_runtime=False)

    def samples(
        self,
        *,
        key_filter: PoolKeyFilter | None = None,
        completion: CompletionFilter = "all",
    ) -> Iterator[PoolSample]:
        return self._store.iter_samples(key_filter=key_filter, completion=completion)

    def samples_list(
        self,
        *,
        key_filter: PoolKeyFilter | None = None,
        completion: CompletionFilter = "all",
    ) -> list[PoolSample]:
        return self._store.bulk_load(key_filter=key_filter, completion=completion)

    def progress(self, *, key_filter: PoolKeyFilter | None = None) -> PoolProgress:
        return self._store.progress(key_filter=key_filter)

    def close(self) -> None:
        if self._closed:
            return
        if self._owns_runtime:
            self._runtime.close()
        self._closed = True

    def __enter__(self) -> PoolReader:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
