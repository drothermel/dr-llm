"""Grid-based seeding for pool samples.

Helpers for seeding the cross-product of N variant axes into a pool's
samples table. Each axis is a named list of "members"; each member has
an id, a value (opaque to dr-llm), and metadata available to callers.

The cross-product walk is generic over request shape: callers supply a
``build_payload`` callback that turns a :class:`GridCell` into the per-row
``PoolSample.request`` dict. For LLM-flavored seeding, see
:func:`seed_llm_grid`, which wraps :func:`seed_grid` with the request
contract :func:`dr_llm.pool.backend.make_llm_process_fn` consumes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import product
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dr_llm.llm import LlmConfig, Message
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.insert_result import InsertResult
from dr_llm.pool.pool_store import PoolStore


class AxisMember[T](BaseModel):
    """One concrete option along a single axis.

    Attributes:
        id: Stable identifier for this member. Becomes part of every
            seeded :attr:`PoolSample.key_values` dict that contains it.
        value: Opaque per-member value handed back to ``build_payload``
            via :class:`GridCell`. dr-llm never inspects it.
        metadata: Self-describing metadata for this member.
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
    used by callers who need a member metadata namespace; defaults to
    ``name`` when omitted.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    members: list[AxisMember[T]]
    metadata_key_prefix: str | None = None

    @property
    def effective_metadata_key_prefix(self) -> str:
        return self.metadata_key_prefix or self.name

    @model_validator(mode="after")
    def _validate_unique_member_ids(self) -> Axis[T]:
        """Reject axes with duplicate AxisMember ids.

        Without this check, two members sharing an id would each generate
        a distinct GridCell with identical ``key_values`` but different
        ``values``. The resulting PoolSample rows would either be
        silently deduped at insert time (under-seeding) or fail against
        the unique ``(key_columns..., sample_idx)`` index on the samples
        table. Failing fast at construction time points the error at the
        actual line declaring the bad axis.
        """
        seen: set[str] = set()
        duplicates: list[str] = []
        for member in self.members:
            if member.id in seen:
                duplicates.append(member.id)
            else:
                seen.add(member.id)
        if duplicates:
            raise ValueError(
                f"Axis {self.name!r} has duplicate AxisMember ids: "
                f"{sorted(set(duplicates))}"
            )
        return self


class GridCell(BaseModel):
    """One cell in the seeded cross-product, passed to ``build_payload``.

    Attributes:
        key_values: Mapping of axis name -> member id. Becomes the
            :attr:`PoolSample.key_values` of the resulting row(s).
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
                name: member.id
                for name, member in zip(axis_names, combo, strict=True)
            },
            values={
                name: member.value
                for name, member in zip(axis_names, combo, strict=True)
            },
        )


def _validate_axes_against_schema(
    store: PoolStore, axes: list[Axis[Any]]
) -> None:
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
    build_metadata: Callable[[GridCell], dict[str, Any]] | None = None,
    chunk_size: int = 500,
) -> InsertResult:
    """Seed the cross-product of ``axes`` into ``store``'s samples table.

    For each cell in the cross-product, calls ``build_payload(cell)`` to
    obtain the row request, then inserts ``n`` :class:`PoolSample` rows
    (one per ``sample_idx`` in ``range(n)``).

    Args:
        store: Target pool store. ``store.schema.key_column_names`` must
            match the axis names in order.
        axes: Ordered list of variant axes. Their ``name`` fields define
            the schema's key columns; their members enumerate the values.
        build_payload: Per-cell callback that returns the
            :attr:`PoolSample.request` dict. Must be JSON-serializable.
        n: Number of samples to seed per cell. Each gets a distinct
            ``sample_idx`` in ``range(n)``.
        build_metadata: Optional per-cell callback for the
            :attr:`PoolSample.metadata` dict. When omitted, per-row
            metadata is empty.
        chunk_size: Maximum number of rows per ``insert_samples`` round-trip.

    Returns:
        Cumulative :class:`InsertResult` summed across all chunks.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    _validate_axes_against_schema(store, axes)

    total = InsertResult()
    chunk: list[PoolSample] = []
    for cell in _iter_cells(axes):
        payload = build_payload(cell)
        metadata = build_metadata(cell) if build_metadata is not None else {}
        for sample_idx in range(n):
            chunk.append(
                PoolSample(
                    key_values=dict(cell.key_values),
                    sample_idx=sample_idx,
                    request=payload,
                    metadata=metadata,
                )
            )
            if len(chunk) >= chunk_size:
                total += store.insert_samples(chunk, ignore_conflicts=True)
                chunk = []
    if chunk:
        total += store.insert_samples(chunk, ignore_conflicts=True)
    return total


# ---------------------------------------------------------------------------
# LLM-specific seeding
# ---------------------------------------------------------------------------


def seed_llm_grid(
    store: PoolStore,
    *,
    axes: list[Axis[Any]],
    build_request: Callable[[GridCell], tuple[list[Message], LlmConfig]],
    n: int = 1,
    build_metadata: Callable[[GridCell], dict[str, Any]] | None = None,
    chunk_size: int = 500,
    llm_config_key: str = "llm_config",
    prompt_key: str = "prompt",
) -> InsertResult:
    """Seed a pool from a cross-product of axes for LLM workers.

    Wraps :func:`seed_grid` with a request shape that
    :func:`dr_llm.pool.backend.make_llm_process_fn` can consume directly:
    each row's request is ``{llm_config_key: <serialized LlmConfig>,
    prompt_key: <list of serialized Messages>}``.

    Args:
        store: Target pool store. Its schema's key columns must match
            the axis names in order.
        axes: Ordered list of variant axes.
        build_request: Per-cell callback returning ``(messages, llm_config)``
            for that cell.
        n: Number of samples per cell (each gets a distinct ``sample_idx``).
        build_metadata: Optional per-cell row-metadata builder.
        chunk_size: Maximum rows per ``insert_samples`` round-trip.
        llm_config_key: Request key for the serialized LlmConfig. Must
            match the value passed to
            :func:`~dr_llm.pool.backend.make_llm_process_fn`.
        prompt_key: Request key for the serialized message list. Must
            match the value passed to
            :func:`~dr_llm.pool.backend.make_llm_process_fn`.

    Returns:
        Cumulative :class:`InsertResult` summed across all chunks.
    """

    def _build_payload(cell: GridCell) -> dict[str, Any]:
        messages, llm_config = build_request(cell)
        return {
            llm_config_key: llm_config.model_dump(mode="json"),
            prompt_key: [
                message.model_dump(mode="json") for message in messages
            ],
        }

    return seed_grid(
        store,
        axes=axes,
        build_payload=_build_payload,
        n=n,
        build_metadata=build_metadata,
        chunk_size=chunk_size,
    )
