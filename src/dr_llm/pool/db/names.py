from __future__ import annotations

from enum import StrEnum


class PoolTableType(StrEnum):
    SAMPLES = "samples"
    CLAIMS = "claims"
    PENDING = "pending"
    METADATA = "metadata"
    CALL_STATS = "call_stats"


class SampleColumn(StrEnum):
    SAMPLE_ID = "sample_id"
    SAMPLE_IDX = "sample_idx"
    PAYLOAD_JSON = "payload_json"
    SOURCE_RUN_ID = "source_run_id"
    METADATA_JSON = "metadata_json"
    CREATED_AT = "created_at"


class ClaimColumn(StrEnum):
    CLAIM_ID = "claim_id"
    RUN_ID = "run_id"
    REQUEST_ID = "request_id"
    CONSUMER_TAG = "consumer_tag"
    SAMPLE_ID = "sample_id"
    CLAIM_IDX = "claim_idx"
    CLAIMED_AT = "claimed_at"


class PendingColumn(StrEnum):
    PENDING_ID = "pending_id"
    SAMPLE_IDX = "sample_idx"
    PAYLOAD_JSON = "payload_json"
    SOURCE_RUN_ID = "source_run_id"
    METADATA_JSON = "metadata_json"
    PRIORITY = "priority"
    STATUS = "status"
    WORKER_ID = "worker_id"
    LEASE_EXPIRES_AT = "lease_expires_at"
    ATTEMPT_COUNT = "attempt_count"
    CREATED_AT = "created_at"


class MetadataColumn(StrEnum):
    POOL_NAME = "pool_name"
    KEY = "key"
    VALUE_JSON = "value_json"
    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"


class CallStatsColumn(StrEnum):
    SAMPLE_ID = "sample_id"
    LATENCY_MS = "latency_ms"
    TOTAL_COST_USD = "total_cost_usd"
    PROMPT_TOKENS = "prompt_tokens"
    COMPLETION_TOKENS = "completion_tokens"
    REASONING_TOKENS = "reasoning_tokens"
    TOTAL_TOKENS = "total_tokens"
    ATTEMPT_COUNT = "attempt_count"
    FINISH_REASON = "finish_reason"
    CREATED_AT = "created_at"
