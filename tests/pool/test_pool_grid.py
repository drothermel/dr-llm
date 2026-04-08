"""Unit tests for grid-based pool seeding helpers."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from dr_llm.pool.db.schema import KeyColumn, PoolSchema
from dr_llm.pool.models import InsertResult
from dr_llm.pool.pending.grid import (
    Axis,
    AxisMember,
    GridCell,
    seed_grid,
)
from dr_llm.pool.pending.pending_sample import PendingSample


def _make_schema() -> PoolSchema:
    return PoolSchema(
        name="grid_test",
        key_columns=[KeyColumn(name="axis_a"), KeyColumn(name="axis_b")],
    )


def _make_store_mock(
    schema: PoolSchema | None = None,
) -> tuple[MagicMock, list[list[PendingSample]], list[tuple[str, dict[str, Any]]]]:
    """Build a stub PoolStore that captures inserts and metadata upserts.

    Returns the mock plus two lists: per-call PendingSample chunks and
    per-call ``(key, value)`` upserts. ``insert_many`` returns an
    ``InsertResult`` reflecting the chunk size, so cumulative totals
    in :func:`seed_grid` match what the real PoolStore would report.
    """
    store = MagicMock()
    store.schema = schema or _make_schema()
    insert_chunks: list[list[PendingSample]] = []
    metadata_upserts: list[tuple[str, dict[str, Any]]] = []

    def _insert_many(
        samples: list[PendingSample], *, ignore_conflicts: bool = True
    ) -> InsertResult:
        insert_chunks.append(list(samples))
        return InsertResult(inserted=len(samples), skipped=0)

    def _upsert(key: str, value: dict[str, Any]) -> None:
        metadata_upserts.append((key, value))

    store.pending.insert_many.side_effect = _insert_many
    store.metadata.upsert.side_effect = _upsert
    return store, insert_chunks, metadata_upserts


def test_axis_member_construction_and_defaults() -> None:
    member = AxisMember[str](id="m1", value="hello")
    assert member.id == "m1"
    assert member.value == "hello"
    assert member.metadata == {}


def test_axis_member_holds_arbitrary_value() -> None:
    class Domain:
        def __init__(self, label: str) -> None:
            self.label = label

    member = AxisMember[Domain](
        id="d1", value=Domain("hi"), metadata={"label": "hi"}
    )
    assert member.value.label == "hi"
    assert member.metadata == {"label": "hi"}


def test_axis_effective_metadata_key_prefix_defaults_to_name() -> None:
    axis = Axis[str](
        name="prompt_template_id",
        members=[AxisMember[str](id="x", value="x")],
    )
    assert axis.effective_metadata_key_prefix == "prompt_template_id"


def test_axis_effective_metadata_key_prefix_override() -> None:
    axis = Axis[str](
        name="prompt_template_id",
        members=[AxisMember[str](id="x", value="x")],
        metadata_key_prefix="prompt_template",
    )
    assert axis.effective_metadata_key_prefix == "prompt_template"


def test_grid_cell_holds_key_values_and_values() -> None:
    cell = GridCell(
        key_values={"axis_a": "1", "axis_b": "2"},
        values={"axis_a": object(), "axis_b": object()},
    )
    assert cell.key_values == {"axis_a": "1", "axis_b": "2"}
    assert set(cell.values.keys()) == {"axis_a", "axis_b"}


def test_seed_grid_inserts_cross_product_with_n_per_cell() -> None:
    store, insert_chunks, _ = _make_store_mock()
    axes: list[Axis[Any]] = [
        Axis(
            name="axis_a",
            members=[
                AxisMember[str](id="a1", value="a1"),
                AxisMember[str](id="a2", value="a2"),
            ],
        ),
        Axis(
            name="axis_b",
            members=[
                AxisMember[str](id="b1", value="b1"),
                AxisMember[str](id="b2", value="b2"),
            ],
        ),
    ]

    result = seed_grid(
        store,
        axes=axes,
        build_payload=lambda cell: {
            "a": cell.values["axis_a"],
            "b": cell.values["axis_b"],
        },
        n=3,
    )

    # 2 x 2 = 4 cells, n=3 -> 12 rows total.
    all_inserted = [s for chunk in insert_chunks for s in chunk]
    assert len(all_inserted) == 12
    assert result.inserted == 12

    cells_seen = {
        (s.key_values["axis_a"], s.key_values["axis_b"]) for s in all_inserted
    }
    assert cells_seen == {
        ("a1", "b1"),
        ("a1", "b2"),
        ("a2", "b1"),
        ("a2", "b2"),
    }

    # Each cell should appear with sample_idx 0..n-1 exactly once.
    for cell_key in cells_seen:
        idxs = sorted(
            s.sample_idx
            for s in all_inserted
            if (s.key_values["axis_a"], s.key_values["axis_b"]) == cell_key
        )
        assert idxs == [0, 1, 2]


def test_seed_grid_upserts_each_axis_member_metadata_once() -> None:
    store, _, upserts = _make_store_mock()
    axes: list[Axis[Any]] = [
        Axis(
            name="axis_a",
            members=[
                AxisMember[str](id="a1", value="a1", metadata={"label": "A1"}),
                AxisMember[str](id="a2", value="a2", metadata={"label": "A2"}),
            ],
        ),
        Axis(
            name="axis_b",
            members=[
                AxisMember[str](id="b1", value="b1", metadata={"label": "B1"}),
            ],
            metadata_key_prefix="dim_b",
        ),
    ]

    seed_grid(
        store,
        axes=axes,
        build_payload=lambda cell: {},
        n=5,
    )

    # 3 axis members total -> 3 metadata upserts (not 3 * 5 cells * 5 samples).
    assert len(upserts) == 3
    upsert_dict = dict(upserts)
    assert upsert_dict["axis_a/a1"] == {"label": "A1"}
    assert upsert_dict["axis_a/a2"] == {"label": "A2"}
    assert upsert_dict["dim_b/b1"] == {"label": "B1"}


def test_seed_grid_chunks_inserts() -> None:
    store, insert_chunks, _ = _make_store_mock()
    axes: list[Axis[Any]] = [
        Axis(
            name="axis_a",
            members=[AxisMember[str](id=f"a{i}", value=f"a{i}") for i in range(5)],
        ),
        Axis(
            name="axis_b",
            members=[AxisMember[str](id=f"b{i}", value=f"b{i}") for i in range(3)],
        ),
    ]
    seed_grid(
        store,
        axes=axes,
        build_payload=lambda cell: {},
        n=2,
        chunk_size=4,
    )
    # 5 * 3 * 2 = 30 rows; chunked at 4 -> ceil(30 / 4) = 8 chunks.
    assert len(insert_chunks) == 8
    assert sum(len(c) for c in insert_chunks) == 30


def test_seed_grid_validates_axis_names_match_schema() -> None:
    store, _, _ = _make_store_mock()
    bad_axes: list[Axis[Any]] = [
        Axis(
            name="wrong",
            members=[AxisMember[str](id="x", value="x")],
        ),
        Axis(
            name="axis_b",
            members=[AxisMember[str](id="y", value="y")],
        ),
    ]
    with pytest.raises(ValueError, match="do not match"):
        seed_grid(
            store, axes=bad_axes, build_payload=lambda cell: {}
        )


def test_seed_grid_rejects_empty_axis() -> None:
    store, _, _ = _make_store_mock()
    axes: list[Axis[Any]] = [
        Axis(name="axis_a", members=[AxisMember[str](id="a", value="a")]),
        Axis(name="axis_b", members=[]),
    ]
    with pytest.raises(ValueError, match="no members"):
        seed_grid(store, axes=axes, build_payload=lambda cell: {})


def test_seed_grid_rejects_empty_axes_list() -> None:
    store, _, _ = _make_store_mock()
    with pytest.raises(ValueError, match="at least one axis"):
        seed_grid(store, axes=[], build_payload=lambda cell: {})


def test_seed_grid_rejects_n_below_one() -> None:
    store, _, _ = _make_store_mock()
    axes: list[Axis[Any]] = [
        Axis(name="axis_a", members=[AxisMember[str](id="a", value="a")]),
        Axis(name="axis_b", members=[AxisMember[str](id="b", value="b")]),
    ]
    with pytest.raises(ValueError, match="n must be"):
        seed_grid(
            store, axes=axes, build_payload=lambda cell: {}, n=0
        )


def test_seed_grid_passes_build_metadata_to_rows() -> None:
    store, insert_chunks, _ = _make_store_mock()
    axes: list[Axis[Any]] = [
        Axis(name="axis_a", members=[AxisMember[str](id="a", value="A")]),
        Axis(name="axis_b", members=[AxisMember[str](id="b", value="B")]),
    ]
    seed_grid(
        store,
        axes=axes,
        build_payload=lambda cell: {"p": cell.key_values},
        build_metadata=lambda cell: {
            "row_label": f"{cell.key_values['axis_a']}-{cell.key_values['axis_b']}"
        },
    )
    sample = insert_chunks[0][0]
    assert sample.metadata == {"row_label": "a-b"}


def test_pool_schema_from_axis_names_builds_text_columns() -> None:
    schema = PoolSchema.from_axis_names(
        "demo", ["axis_a", "axis_b", "axis_c"]
    )
    assert schema.name == "demo"
    assert schema.key_column_names == ["axis_a", "axis_b", "axis_c"]
    assert all(kc.type.value == "text" for kc in schema.key_columns)


def test_pool_schema_from_axis_names_validates_names() -> None:
    with pytest.raises(ValueError, match="lowercase"):
        PoolSchema.from_axis_names("demo", ["BadName"])
