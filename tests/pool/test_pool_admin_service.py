"""Unit tests for pool admin models and request validation."""

from __future__ import annotations

from dr_llm.pool.admin.creation import (
    CreatePoolRequest,
    PoolCreationBlockReason,
    _request_violations,
)
from dr_llm.pool.admin.deletion import (
    DeletePoolRequest,
    PoolDeletionBlockReason,
    _claim_table_belongs_to_pool,
    _pool_delete_request_violations,
)
from dr_llm.pool.admin.inspection import PoolInspection
from dr_llm.pool.db.schema import PoolSchema, KeyColumn
from dr_llm.pool.pool_progress import PoolProgress


def test_pool_inspection_with_progress() -> None:
    progress = PoolProgress(total=10, incomplete=3, leased=1, complete=7, error=2)
    inspection = PoolInspection(
        project_name="proj",
        name="my_pool",
        pool_schema=PoolSchema(name="my_pool", key_columns=[KeyColumn(name="dim")]),
        progress=progress,
    )
    assert inspection.progress.total == 10
    assert inspection.progress.incomplete == 3
    assert inspection.progress.error == 2


def test_create_pool_request_validates_pool_name() -> None:
    request = CreatePoolRequest(project_name="proj", pool_name="INVALID NAME!")
    violations = _request_violations(request)
    assert any(
        v.reason == PoolCreationBlockReason.invalid_pool_name for v in violations
    )


def test_create_pool_request_requires_key_axes() -> None:
    request = CreatePoolRequest(project_name="proj", pool_name="valid_pool")
    violations = _request_violations(request)
    assert any(v.reason == PoolCreationBlockReason.missing_key_axes for v in violations)


def test_delete_pool_request_validates_pool_name() -> None:
    request = DeletePoolRequest(project_name="proj", pool_name="BAD NAME!")
    violations = _pool_delete_request_violations(request)
    assert any(
        v.reason == PoolDeletionBlockReason.invalid_pool_name for v in violations
    )


def test_claim_table_match_uses_most_specific_pool_name() -> None:
    known_pool_names = {"foo", "foo_claims_bar"}
    table_name = "pool_foo_claims_bar_claims_baz"

    assert (
        _claim_table_belongs_to_pool(
            table_name,
            pool_name="foo",
            known_pool_names=known_pool_names,
        )
        is False
    )
    assert (
        _claim_table_belongs_to_pool(
            table_name,
            pool_name="foo_claims_bar",
            known_pool_names=known_pool_names,
        )
        is True
    )


def test_claim_table_match_excludes_known_pool_tables() -> None:
    known_pool_names = {"foo", "foo_claims_bar"}

    assert (
        _claim_table_belongs_to_pool(
            "pool_foo_claims_bar_samples",
            pool_name="foo",
            known_pool_names=known_pool_names,
        )
        is False
    )
