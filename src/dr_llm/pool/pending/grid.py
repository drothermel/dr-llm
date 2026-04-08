"""Grid-based seeding for pool pending queues.

Helpers for seeding the cross-product of N variant axes into a pool's
pending queue. Each axis is a named list of "members"; each member has
an id, a value (opaque to dr-llm), and metadata that gets upserted to
the pool's :class:`MetadataStore` exactly once per seed call.

The cross-product walk is generic over payload shape: callers supply a
``build_payload`` callback that turns a :class:`GridCell` into the per-row
``PendingSample.payload`` dict. For LLM-flavored seeding, see
:func:`dr_llm.pool.llm_pool_adapter.seed_llm_grid`, which wraps this
function with the payload contract :func:`make_llm_process_fn` consumes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import product
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.pool.models import InsertResult
from dr_llm.pool.pending.pending_sample import PendingSample
from dr_llm.pool.pool_store import PoolStore


class AxisMember[T](BaseModel):
    """One concrete option along a single axis.

    Attributes:
        id: Stable identifier for this member. Becomes part of every
            seeded :attr:`PendingSample.key_values` dict that contains it.
        value: Opaque per-member value handed back to ``build_payload``
            via :class:`GridCell`. dr-llm never inspects it.
        metadata: Self-describing metadata for this member. Upserted
            once per :func:`seed_grid` call into the pool's
            :class:`MetadataStore`, keyed by
            ``f"{axis.metadata_key_prefix or axis.name}/{id}"``.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    id: str
    value: T
    metadata: dict[str, Any] = Field(default_factory=dict)


class Axis[T](BaseModel):
    """A named axis of variants for cross-product seeding.

    The ``name`` becomes a key column in the pool schema (callers can
    use :meth:`PoolSchema.from_axis_names` to derive the schema from a
    list of axis names). ``metadata_key_prefix`` controls the namespace
    used when upserting member metadata; defaults to ``name`` when omitted.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    members: list[AxisMember[T]]
    metadata_key_prefix: str | None = None

    @property
    def effective_metadata_key_prefix(self) -> str:
        return self.metadata_key_prefix or self.name


class GridCell(BaseModel):
    """One cell in the seeded cross-product, passed to ``build_payload``.

    Attributes:
        key_values: Mapping of axis name -> member id. Becomes the
            :attr:`PendingSample.key_values` of the resulting row(s).
        values: Mapping of axis name -> :attr:`AxisMember.value`. The
            opaque per-member values, ready for use in payload assembly.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    key_values: dict[str, str]
    values: dict[str, Any]


def _iter_cells(axes: list[Axis[Any]]) -> Iterator[GridCell]:
    axis_names = [axis.name for axis in axes]
    for combo in product(*(axis.members for axis in axes)):
        yield GridCell(
            key_values={
                name: member.id for name, member in zip(axis_names, combo, strict=True)
            },
            values={
                name: member.value
                for name, member in zip(axis_names, combo, strict=True)
            },
        )


def _validate_axes_against_schema(store: PoolStore, axes: list[Axis[Any]]) -> None:
    if not axes:
        raise ValueError("seed_grid requires at least one axis")
    schema_names = store.schema.key_column_names
    axis_names = [axis.name for axis in axes]
    if axis_names != schema_names:
        raise ValueError(
            "Axis names do not match the pool schema's key columns. "
            f"axes={axis_names!r} schema={schema_names!r}"
        )
    for axis in axes:
        if not axis.members:
            raise ValueError(f"Axis {axis.name!r} has no members")


def seed_grid(
    store: PoolStore,
    *,
    axes: list[Axis[Any]],
    build_payload: Callable[[GridCell], dict[str, Any]],
    n: int = 1,
    priority: int = 0,
    build_metadata: Callable[[GridCell], dict[str, Any]] | None = None,
    chunk_size: int = 500,
) -> InsertResult:
    """Seed the cross-product of ``axes`` into ``store``'s pending queue.

    For each cell in the cross-product, calls ``build_payload(cell)`` to
    obtain the row payload, then inserts ``n`` :class:`PendingSample` rows
    (one per ``sample_idx`` in ``range(n)``). Each axis member's metadata
    is upserted to the pool's :class:`MetadataStore` exactly once.

    Args:
        store: Target pool store. ``store.schema.key_column_names`` must
            match the axis names in order.
        axes: Ordered list of variant axes. Their ``name`` fields define
            the schema's key columns; their members enumerate the values.
        build_payload: Per-cell callback that returns the
            :attr:`PendingSample.payload` dict. Must be JSON-serializable.
        n: Number of samples to seed per cell. Each gets a distinct
            ``sample_idx`` in ``range(n)``.
        priority: Priority assigned to all seeded rows.
        build_metadata: Optional per-cell callback for the
            :attr:`PendingSample.metadata` dict. When omitted, per-row
            metadata is empty (axis-member metadata is still upserted
            to the pool's :class:`MetadataStore`).
        chunk_size: Maximum number of rows per ``insert_many`` round-trip.

    Returns:
        Cumulative :class:`InsertResult` summed across all chunks.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    _validate_axes_against_schema(store, axes)

    for axis in axes:
        prefix = axis.effective_metadata_key_prefix
        for member in axis.members:
            store.metadata.upsert(
                f"{prefix}/{member.id}",
                member.metadata,
            )

    total = InsertResult()
    chunk: list[PendingSample] = []
    for cell in _iter_cells(axes):
        payload = build_payload(cell)
        metadata = build_metadata(cell) if build_metadata is not None else {}
        for sample_idx in range(n):
            chunk.append(
                PendingSample(
                    key_values=dict(cell.key_values),
                    sample_idx=sample_idx,
                    priority=priority,
                    payload=payload,
                    metadata=metadata,
                )
            )
            if len(chunk) >= chunk_size:
                total += store.pending.insert_many(chunk, ignore_conflicts=True)
                chunk = []
    if chunk:
        total += store.pending.insert_many(chunk, ignore_conflicts=True)
    return total
