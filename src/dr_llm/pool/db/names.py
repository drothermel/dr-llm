from __future__ import annotations

from enum import StrEnum


class PoolTableType(StrEnum):
    SAMPLES = "samples"
    LEASES = "leases"


class IndexNamePrefix(StrEnum):
    UNIQUE = "uq"
    STANDARD = "idx"


class PoolIndexName(StrEnum):
    CELL = "cell"
    KEY = "key"
    INCOMPLETE = "incomplete"


def pool_index_name(
    prefix: IndexNamePrefix, table_name: str, index_name: PoolIndexName
) -> str:
    return f"{prefix}_{table_name}_{index_name}"


class SampleColumn(StrEnum):
    SAMPLE_ID = "sample_id"
    SAMPLE_IDX = "sample_idx"
    RUN_ID = "run_id"
    REQUEST_JSON = "request_json"
    RESPONSE_JSON = "response_json"
    FINISH_REASON = "finish_reason"
    ATTEMPT_COUNT = "attempt_count"
    METADATA_JSON = "metadata_json"
    CREATED_AT = "created_at"


class LeaseColumn(StrEnum):
    SAMPLE_ID = "sample_id"
    WORKER_ID = "worker_id"
    LEASE_EXPIRES_AT = "lease_expires_at"
