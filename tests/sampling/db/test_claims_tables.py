"""Unit tests for sampling claims table metadata."""

from __future__ import annotations

from dr_llm.sampling.db.claims_tables import ClaimsTables
from dr_llm.sampling.db.names import (
    ClaimsIndexName,
    IndexNamePrefix,
    claims_index_name,
    claims_table_name,
)


def test_claims_table_name() -> None:
    assert (
        claims_table_name("mypool", "sweep_01")
        == "pool_mypool_claims_sweep_01"
    )


def test_claims_tables_builds_table() -> None:
    ct = ClaimsTables("mypool", "sweep_01")
    assert ct.table_name == "pool_mypool_claims_sweep_01"
    assert ct.claims_table.name == "pool_mypool_claims_sweep_01"

    column_names = [c.name for c in ct.claims_table.columns]
    assert "claim_id" in column_names
    assert "run_id" in column_names
    assert "sample_id" in column_names
    assert "claim_idx" in column_names
    assert "claimed_at" in column_names


def test_claims_tables_builds_indexes() -> None:
    ct = ClaimsTables("mypool", "sweep_01")
    index_names = {idx.name for idx in ct.claims_table.indexes}
    expected_uq = claims_index_name(
        IndexNamePrefix.UNIQUE,
        ct.table_name,
        ClaimsIndexName.RUN_SAMPLE,
    )
    expected_idx = claims_index_name(
        IndexNamePrefix.STANDARD,
        ct.table_name,
        ClaimsIndexName.RUN,
    )
    assert expected_uq in index_names
    assert expected_idx in index_names


def test_claims_index_name() -> None:
    assert (
        claims_index_name(
            IndexNamePrefix.UNIQUE,
            "pool_test_claims_sweep",
            ClaimsIndexName.RUN_SAMPLE,
        )
        == "uq_pool_test_claims_sweep_run_sample"
    )
