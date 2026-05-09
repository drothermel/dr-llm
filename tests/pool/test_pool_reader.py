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
