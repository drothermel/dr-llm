from __future__ import annotations

from typing import Any, cast

import pandas as pd

from dr_llm.pool.db.schema import PoolSchema
from dr_llm.pool.pending.pending_status import PendingStatus


_CALL_STATS_COLUMNS = [
    "latency_ms",
    "total_cost_usd",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "total_tokens",
    "attempt_count",
    "finish_reason",
    "call_stats_created_at",
]
_PENDING_COLUMNS = [
    "pending_id",
    "pending_status",
    "pending_worker_id",
    "pending_priority",
    "pending_attempt_count",
    "pending_created_at",
    "pending_lease_expires_at",
    "fail_reason",
]
_HEAVY_METADATA_FIELDS = {"source_code", "template_text"}


def build_pool_data_frame(
    *,
    schema: PoolSchema,
    samples_frame: pd.DataFrame,
    pending_frame: pd.DataFrame,
    metadata_frame: pd.DataFrame,
    call_stats_frame: pd.DataFrame,
    include_raw: bool = True,
    include_failed: bool = True,
) -> pd.DataFrame:
    """Build a pool dataframe with prompt, result, stats, axes, and failures."""
    key_columns = schema.key_column_names
    successes = _build_success_outcome_frame(
        samples_frame=samples_frame,
        pending_frame=pending_frame,
        call_stats_frame=call_stats_frame,
        key_columns=key_columns,
        include_raw=True,
    )
    frames = [successes]
    if include_failed:
        frames.append(
            _build_failure_outcome_frame(
                pending_frame=pending_frame,
                key_columns=key_columns,
                include_raw=True,
            )
        )
    frame = pd.concat(frames, ignore_index=True, sort=False)
    frame = _attach_axis_metadata(
        frame,
        metadata_frame=metadata_frame,
        key_columns=key_columns,
        include_heavy=include_raw,
    )
    if not include_raw:
        frame = _drop_raw_columns(frame)
    return _sort_frame(frame, [*key_columns, "sample_idx", "outcome"])


def _build_success_outcome_frame(
    *,
    samples_frame: pd.DataFrame,
    pending_frame: pd.DataFrame,
    call_stats_frame: pd.DataFrame,
    key_columns: list[str],
    include_raw: bool,
) -> pd.DataFrame:
    samples = _prepare_samples_frame(samples_frame, include_raw=include_raw)
    if samples.empty:
        return samples

    call_stats = _prepare_call_stats_frame(call_stats_frame)
    frame = samples.merge(call_stats, on="sample_id", how="left")
    frame["outcome"] = "success_missing_call_stats"
    frame.loc[frame["latency_ms"].notna(), "outcome"] = "success"

    promoted = _prepare_pending_frame(pending_frame, include_raw=include_raw)
    promoted = promoted.loc[
        promoted["pending_status"] == PendingStatus.promoted.value
    ].copy()
    pending_columns = [
        column
        for column in [
            *key_columns,
            "sample_idx",
            *_PENDING_COLUMNS,
            "prompt_messages",
            "prompt_text",
            "pending_payload_json",
            "pending_metadata_json",
        ]
        if column in promoted.columns
    ]
    if pending_columns:
        frame = frame.merge(
            promoted.loc[:, pending_columns],
            on=[*key_columns, "sample_idx"],
            how="left",
        )
    frame["result_text"] = frame["result_payload_json"].map(_extract_result_text)
    return frame


def _build_failure_outcome_frame(
    *,
    pending_frame: pd.DataFrame,
    key_columns: list[str],
    include_raw: bool,
) -> pd.DataFrame:
    pending = _prepare_pending_frame(pending_frame, include_raw=include_raw)
    if pending.empty:
        return pending

    failures = pending.loc[
        pending["pending_status"] == PendingStatus.failed.value
    ].copy()
    if failures.empty:
        return failures

    failures["sample_id"] = pd.NA
    failures["outcome"] = "failed"
    failures["result_text"] = pd.NA
    for column in _CALL_STATS_COLUMNS:
        failures[column] = pd.NA
    if not include_raw:
        failures = failures.drop(
            columns=[
                column
                for column in ["pending_payload_json", "pending_metadata_json"]
                if column in failures.columns
            ]
        )
    return failures


def _drop_raw_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(
        columns=[
            column
            for column in [
                "result_payload_json",
                "sample_metadata_json",
                "prompt_messages",
                "pending_payload_json",
                "pending_metadata_json",
            ]
            if column in frame.columns
        ]
    )


def _prepare_samples_frame(
    samples_frame: pd.DataFrame,
    *,
    include_raw: bool,
) -> pd.DataFrame:
    frame = samples_frame.copy().rename(
        columns={
            "created_at": "sample_created_at",
            "payload_json": "result_payload_json",
            "metadata_json": "sample_metadata_json",
            "source_run_id": "sample_source_run_id",
        }
    )
    if not include_raw:
        frame = frame.drop(
            columns=[
                column
                for column in ["result_payload_json", "sample_metadata_json"]
                if column in frame.columns
            ]
        )
    return frame


def _prepare_call_stats_frame(call_stats_frame: pd.DataFrame) -> pd.DataFrame:
    frame = call_stats_frame.copy()
    if frame.empty and "sample_id" not in frame.columns:
        return pd.DataFrame(columns=["sample_id", *_CALL_STATS_COLUMNS])
    return frame.rename(columns={"created_at": "call_stats_created_at"})


def _prepare_pending_frame(
    pending_frame: pd.DataFrame,
    *,
    include_raw: bool,
) -> pd.DataFrame:
    frame = pending_frame.copy().rename(
        columns={
            "status": "pending_status",
            "worker_id": "pending_worker_id",
            "priority": "pending_priority",
            "attempt_count": "pending_attempt_count",
            "created_at": "pending_created_at",
            "lease_expires_at": "pending_lease_expires_at",
            "payload_json": "pending_payload_json",
            "metadata_json": "pending_metadata_json",
        }
    )
    if frame.empty:
        return frame
    frame["fail_reason"] = frame["pending_metadata_json"].map(_extract_fail_reason)
    frame["prompt_messages"] = frame["pending_payload_json"].map(_extract_prompt)
    frame["prompt_text"] = frame["prompt_messages"].map(_messages_to_text)
    if not include_raw:
        frame = frame.drop(
            columns=[
                column
                for column in [
                    "pending_payload_json",
                    "pending_metadata_json",
                    "prompt_messages",
                    "prompt_text",
                ]
                if column in frame.columns
            ]
        )
    return frame


def _attach_axis_metadata(
    frame: pd.DataFrame,
    *,
    metadata_frame: pd.DataFrame,
    key_columns: list[str],
    include_heavy: bool,
) -> pd.DataFrame:
    if frame.empty or metadata_frame.empty:
        return frame
    result = frame
    for key_column in key_columns:
        dimension = _metadata_dimension_frame(
            metadata_frame,
            key_column=key_column,
            include_heavy=include_heavy,
        )
        if not dimension.empty:
            result = result.merge(dimension, on=key_column, how="left")
    return result


def _metadata_dimension_frame(
    metadata_frame: pd.DataFrame,
    *,
    key_column: str,
    include_heavy: bool,
) -> pd.DataFrame:
    if (
        "key" not in metadata_frame.columns
        or "value_json" not in metadata_frame.columns
    ):
        return pd.DataFrame(columns=[key_column])

    prefix = _metadata_prefix_for_key_column(metadata_frame, key_column)
    if prefix is None:
        return pd.DataFrame(columns=[key_column])

    key_prefix = f"{prefix}/"
    rows: list[dict[str, Any]] = []
    for row in metadata_frame.loc[
        metadata_frame["key"].astype("string").str.startswith(key_prefix, na=False)
    ].to_dict(orient="records"):
        metadata_key = str(row["key"])
        member_id = metadata_key.removeprefix(key_prefix)
        value = row.get("value_json")
        record = {key_column: member_id}
        record.update(
            _flatten_metadata_value(
                prefix,
                value,
                include_heavy=include_heavy,
            )
        )
        rows.append(record)

    if not rows:
        return pd.DataFrame(columns=[key_column])
    return pd.DataFrame.from_records(rows).drop_duplicates(
        subset=[key_column],
        keep="last",
    )


def _metadata_prefix_for_key_column(
    metadata_frame: pd.DataFrame,
    key_column: str,
) -> str | None:
    candidates = [key_column]
    if key_column.endswith("_id"):
        candidates.append(key_column.removesuffix("_id"))

    metadata_keys = metadata_frame["key"].astype("string")
    for candidate in candidates:
        if metadata_keys.str.startswith(f"{candidate}/", na=False).any():
            return candidate
    return None


def _flatten_metadata_value(
    prefix: str,
    value: object,
    *,
    include_heavy: bool,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    fields: dict[str, Any] = {}
    value_dict = cast(dict[str, Any], value)
    for key, item in value_dict.items():
        if key == prefix and isinstance(item, dict):
            fields.update(
                _flatten_scalar_tree(
                    prefix,
                    cast(dict[str, Any], item),
                    include_heavy=include_heavy,
                )
            )
            continue
        if key == "block_ids_by_category" and isinstance(item, dict):
            fields.update(_flatten_block_ids(prefix, cast(dict[str, Any], item)))
            continue
        if _is_scalar(item) and _include_metadata_field(key, include_heavy):
            fields[f"{prefix}__{key}"] = item
    return fields


def _flatten_scalar_tree(
    prefix: str,
    value: dict[str, Any],
    *,
    include_heavy: bool,
    path: tuple[str, ...] = (),
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key, item in value.items():
        next_path = (*path, key)
        if isinstance(item, dict):
            fields.update(
                _flatten_scalar_tree(
                    prefix,
                    item,
                    include_heavy=include_heavy,
                    path=next_path,
                )
            )
        elif _is_scalar(item) and _include_metadata_field(key, include_heavy):
            fields[f"{prefix}__{'__'.join(next_path)}"] = item
    return fields


def _flatten_block_ids(prefix: str, value: dict[str, Any]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for category, members in value.items():
        if isinstance(members, list):
            fields[f"{prefix}__{category}"] = "|".join(
                str(member) for member in members
            )
        elif _is_scalar(members):
            fields[f"{prefix}__{category}"] = str(members)
    return fields


def _include_metadata_field(key: str, include_heavy: bool) -> bool:
    return include_heavy or key not in _HEAVY_METADATA_FIELDS


def _extract_fail_reason(value: object) -> object:
    if isinstance(value, dict):
        return cast(dict[str, Any], value).get("fail_reason", "")
    return ""


def _extract_prompt(value: object) -> object:
    if isinstance(value, dict):
        return cast(dict[str, Any], value).get("prompt")
    return pd.NA


def _extract_result_text(value: object) -> object:
    if not isinstance(value, dict):
        return pd.NA
    value_dict = cast(dict[str, Any], value)
    for key in ["text", "result", "content"]:
        candidate = value_dict.get(key)
        if isinstance(candidate, str):
            return candidate
    return pd.NA


def _messages_to_text(value: object) -> object:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return pd.NA
    chunks = [_message_content_text(item) for item in value]
    text = "\n\n".join(chunk for chunk in chunks if chunk)
    return text if text else pd.NA


def _message_content_text(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    value_dict = cast(dict[str, Any], value)
    content = value_dict.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return ""


def _is_scalar(value: object) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def _sort_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    sort_columns = [column for column in columns if column in frame.columns]
    if frame.empty or not sort_columns:
        return frame.reset_index(drop=True)
    return frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)
