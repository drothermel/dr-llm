"""Unit tests for PoolReader pure-Python pieces."""

from __future__ import annotations

import pytest

from dr_llm.pool.pending.pending_status import PendingStatusCounts
from dr_llm.pool.reader import PoolProgress, _validate_pool_name


def test_pool_progress_in_flight_delegates_to_pending_counts() -> None:
    progress = PoolProgress(
        samples_total=10,
        pending_counts=PendingStatusCounts(pending=3, leased=2, failed=1),
    )
    assert progress.in_flight == 5
    assert progress.is_complete is False


def test_pool_progress_is_complete_when_no_in_flight() -> None:
    progress = PoolProgress(
        samples_total=10,
        pending_counts=PendingStatusCounts(pending=0, leased=0, failed=2),
    )
    assert progress.in_flight == 0
    assert progress.is_complete is True


def test_pool_progress_computed_fields_appear_in_model_dump() -> None:
    """Derived properties must be exposed via @computed_field so they
    survive model_dump / JSON serialization."""
    progress = PoolProgress(
        samples_total=42,
        pending_counts=PendingStatusCounts(pending=1, leased=2, failed=0),
    )
    dumped = progress.model_dump()
    assert dumped["samples_total"] == 42
    assert dumped["in_flight"] == 3
    assert dumped["is_complete"] is False
    # pending_counts is a nested model; its own derived fields are also
    # subject to whether they were declared with @computed_field. The
    # existing PendingStatusCounts uses plain @property so its in_flight
    # is NOT in the dump — that's out of scope for this PR.
    assert "pending_counts" in dumped


def test_pool_progress_is_frozen() -> None:
    progress = PoolProgress(
        samples_total=1,
        pending_counts=PendingStatusCounts(),
    )
    with pytest.raises(Exception, match="frozen"):
        progress.samples_total = 99


@pytest.mark.parametrize(
    "name",
    ["a", "abc", "a1", "snake_case", "with_123_digits"],
)
def test_validate_pool_name_accepts_valid(name: str) -> None:
    _validate_pool_name(name)  # should not raise


@pytest.mark.parametrize(
    "name",
    [
        "",
        "1starts_with_digit",
        "_underscore_start",
        "Capital",
        "has-dash",
        "has.dot",
        "has space",
        "has;semicolon",  # SQL injection style
        'has"quote',
    ],
)
def test_validate_pool_name_rejects_invalid(name: str) -> None:
    with pytest.raises(ValueError, match="pool_name must be"):
        _validate_pool_name(name)
