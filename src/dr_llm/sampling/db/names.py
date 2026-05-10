from __future__ import annotations

from enum import StrEnum


class ClaimsTableType(StrEnum):
    CLAIMS = "claims"


class ClaimColumn(StrEnum):
    CLAIM_ID = "claim_id"
    RUN_ID = "run_id"
    REQUEST_ID = "request_id"
    CONSUMER_TAG = "consumer_tag"
    SAMPLE_ID = "sample_id"
    CLAIM_IDX = "claim_idx"
    CLAIMED_AT = "claimed_at"


class IndexNamePrefix(StrEnum):
    UNIQUE = "uq"
    STANDARD = "idx"


class ClaimsIndexName(StrEnum):
    RUN_SAMPLE = "run_sample"
    RUN = "run"


def claims_index_name(
    prefix: IndexNamePrefix, table_name: str, index_name: ClaimsIndexName
) -> str:
    return f"{prefix}_{table_name}_{index_name}"


def claims_table_name(pool_name: str, consumer_id: str) -> str:
    return f"pool_{pool_name}_claims_{consumer_id}"
