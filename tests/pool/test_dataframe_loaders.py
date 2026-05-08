from __future__ import annotations

import pandas as pd

from dr_llm.pool.dataframe_loaders import build_pool_data_frame
from dr_llm.pool.db.schema import KeyColumn, PoolSchema


_SCHEMA = PoolSchema(
    name="loader_test",
    key_columns=[
        KeyColumn(name="prompt_template_id"),
        KeyColumn(name="data_sample_id"),
        KeyColumn(name="llm_config_id"),
    ],
)


def _samples_frame() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "sample_id": "sample-success",
                "prompt_template_id": "prompt-a",
                "data_sample_id": "data-a",
                "llm_config_id": "config-a",
                "sample_idx": 0,
                "payload_json": {"text": "generated answer"},
                "source_run_id": "run-1",
                "metadata_json": {"kind": "complete"},
                "created_at": "2026-01-01T00:00:00Z",
            },
            {
                "sample_id": "sample-missing-stats",
                "prompt_template_id": "prompt-a",
                "data_sample_id": "data-a",
                "llm_config_id": "config-a",
                "sample_idx": 1,
                "payload_json": {"text": "second answer"},
                "source_run_id": "run-1",
                "metadata_json": {},
                "created_at": "2026-01-01T00:01:00Z",
            },
        ]
    )


def _pending_frame() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "pending_id": "pending-success",
                "prompt_template_id": "prompt-a",
                "data_sample_id": "data-a",
                "llm_config_id": "config-a",
                "sample_idx": 0,
                "payload_json": {
                    "prompt": [{"role": "user", "content": "Describe this code"}],
                    "llm_config": {"provider": "openai", "model": "gpt-test"},
                },
                "source_run_id": "run-1",
                "metadata_json": {},
                "priority": 10,
                "status": "promoted",
                "worker_id": "worker-1",
                "lease_expires_at": None,
                "attempt_count": 1,
                "created_at": "2026-01-01T00:00:00Z",
            },
            {
                "pending_id": "pending-failed",
                "prompt_template_id": "prompt-a",
                "data_sample_id": "data-a",
                "llm_config_id": "config-b",
                "sample_idx": 0,
                "payload_json": {
                    "prompt": [{"role": "user", "content": "Failing prompt"}],
                    "llm_config": {"provider": "anthropic", "model": "claude-test"},
                },
                "source_run_id": "run-2",
                "metadata_json": {"fail_reason": "RateLimitError: busy"},
                "priority": 5,
                "status": "failed",
                "worker_id": "worker-2",
                "lease_expires_at": "2026-01-01T00:02:00Z",
                "attempt_count": 2,
                "created_at": "2026-01-01T00:01:00Z",
            },
        ]
    )


def _metadata_frame() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "pool_name": "loader_test",
                "key": "prompt_template/prompt-a",
                "value_json": {
                    "template_text": "Prompt template {{CODE}}",
                    "block_ids_by_category": {
                        "task": ["describe_functionality"],
                        "goal": ["usable_for_reconstruction"],
                    },
                },
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            },
            {
                "pool_name": "loader_test",
                "key": "data_sample/data-a",
                "value_json": {
                    "task_id": "HumanEval/1",
                    "dataset_id": "human_eval",
                    "source_code": "def f(): ...",
                },
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            },
            {
                "pool_name": "loader_test",
                "key": "llm_config/config-a",
                "value_json": {
                    "llm_config": {
                        "provider": "openai",
                        "model": "gpt-test",
                        "reasoning": {"thinking_level": "low"},
                    }
                },
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            },
            {
                "pool_name": "loader_test",
                "key": "llm_config/config-b",
                "value_json": {
                    "llm_config": {
                        "provider": "anthropic",
                        "model": "claude-test",
                        "reasoning": {"thinking_level": "medium"},
                    }
                },
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            },
        ]
    )


def _call_stats_frame() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "sample_id": "sample-success",
                "latency_ms": 100,
                "total_cost_usd": 0.01,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "reasoning_tokens": None,
                "total_tokens": 15,
                "attempt_count": 1,
                "finish_reason": "stop",
                "created_at": "2026-01-01T00:00:10Z",
            }
        ]
    )


def test_pool_data_frame_returns_prompt_result_stats_and_failures() -> None:
    frame = build_pool_data_frame(
        schema=_SCHEMA,
        samples_frame=_samples_frame(),
        pending_frame=_pending_frame(),
        metadata_frame=_metadata_frame(),
        call_stats_frame=_call_stats_frame(),
    )

    assert set(frame["outcome"]) == {
        "success",
        "success_missing_call_stats",
        "failed",
    }
    failed = frame.loc[frame["outcome"] == "failed"].iloc[0]
    assert failed["fail_reason"] == "RateLimitError: busy"
    assert failed["llm_config__provider"] == "anthropic"
    assert failed["prompt_text"] == "Failing prompt"
    assert pd.isna(failed["latency_ms"])
    assert pd.isna(failed["result_text"])

    success = frame.loc[frame["outcome"] == "success"].iloc[0]
    assert success["prompt_text"] == "Describe this code"
    assert success["result_text"] == "generated answer"
    assert success["prompt_template__task"] == "describe_functionality"
    assert success["prompt_template__goal"] == "usable_for_reconstruction"
    assert success["data_sample__task_id"] == "HumanEval/1"
    assert success["llm_config__reasoning__thinking_level"] == "low"
    assert "result_payload_json" in frame.columns
    assert "pending_payload_json" in frame.columns
    assert "prompt_messages" in frame.columns
    assert "prompt_template__template_text" in frame.columns
    assert "data_sample__source_code" in frame.columns


def test_pool_data_frame_can_omit_failed_rows() -> None:
    frame = build_pool_data_frame(
        schema=_SCHEMA,
        samples_frame=_samples_frame(),
        pending_frame=_pending_frame(),
        metadata_frame=_metadata_frame(),
        call_stats_frame=_call_stats_frame(),
        include_failed=False,
    )

    assert set(frame["outcome"]) == {"success", "success_missing_call_stats"}


def test_pool_data_frame_can_drop_raw_fields_but_keep_text() -> None:
    frame = build_pool_data_frame(
        schema=_SCHEMA,
        samples_frame=_samples_frame(),
        pending_frame=_pending_frame(),
        metadata_frame=_metadata_frame(),
        call_stats_frame=_call_stats_frame(),
        include_raw=False,
    )

    first = frame.loc[frame["sample_id"] == "sample-success"].iloc[0]
    assert first["prompt_text"] == "Describe this code"
    assert first["result_text"] == "generated answer"
    assert first["latency_ms"] == 100
    assert first["llm_config__provider"] == "openai"
    assert "result_payload_json" not in frame.columns
    assert "pending_payload_json" not in frame.columns
    assert "prompt_messages" not in frame.columns
    assert "prompt_template__template_text" not in frame.columns
    assert "data_sample__source_code" not in frame.columns
