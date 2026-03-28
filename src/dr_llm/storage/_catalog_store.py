from __future__ import annotations

import json
from typing import Any, Literal
from uuid import uuid4

from psycopg.rows import dict_row
from psycopg.sql import SQL

from dr_llm.errors import PersistenceError
from dr_llm.storage._runtime import StorageRuntime
from dr_llm.types import (
    ModelCatalogEntry,
    ModelCatalogPricing,
    ModelCatalogQuery,
    ModelCatalogRateLimit,
)


class CatalogStore:
    def __init__(self, runtime: StorageRuntime) -> None:
        self._runtime = runtime

    def _model_filter_clause(
        self, query: ModelCatalogQuery
    ) -> tuple[SQL, list[object]]:
        filter_clause = SQL("""
                WHERE (%s::text IS NULL OR provider = %s)
                  AND (%s::boolean IS NULL OR supports_reasoning = %s)
                  AND (%s::text IS NULL OR model ILIKE %s)
        """)
        filter_params: list[object] = [
            query.provider,
            query.provider,
            query.supports_reasoning,
            query.supports_reasoning,
            query.model_contains,
            f"%{query.model_contains}%" if query.model_contains is not None else None,
        ]
        return filter_clause, filter_params

    def record_catalog_snapshot(
        self,
        *,
        provider: str,
        status: str,
        raw_payload: dict[str, Any] | None = None,
        error_text: str | None = None,
    ) -> str:
        self._runtime.init_schema()
        snapshot_id = uuid4().hex
        with self._runtime.conn() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO provider_model_catalog_snapshots (
                        snapshot_id,
                        provider,
                        fetched_at,
                        status,
                        raw_json,
                        error_text
                    ) VALUES (%s, %s, now(), %s, %s::jsonb, %s)
                    """,
                    [
                        snapshot_id,
                        provider,
                        status,
                        json.dumps(raw_payload or {}, ensure_ascii=True),
                        error_text,
                    ],
                )
            except Exception as exc:
                conn.rollback()
                raise PersistenceError(
                    f"Failed to record model catalog snapshot: {exc}"
                ) from exc
            else:
                conn.commit()
                return snapshot_id

    def replace_provider_models(
        self,
        *,
        provider: str,
        entries: list[ModelCatalogEntry],
    ) -> int:
        self._runtime.init_schema()
        with self._runtime.conn() as conn:
            try:
                conn.execute(
                    "DELETE FROM provider_models_current WHERE provider = %s",
                    [provider],
                )
                if entries:
                    params = [
                        [
                            entry.provider,
                            entry.model,
                            entry.display_name,
                            entry.context_window,
                            entry.max_output_tokens,
                            entry.supports_reasoning,
                            entry.supports_tools,
                            entry.supports_vision,
                            json.dumps(
                                (
                                    entry.pricing.model_dump(
                                        mode="json",
                                        exclude_none=True,
                                        exclude_computed_fields=True,
                                    )
                                    if entry.pricing is not None
                                    else {}
                                ),
                                ensure_ascii=True,
                            ),
                            json.dumps(
                                (
                                    entry.rate_limits.model_dump(
                                        mode="json",
                                        exclude_none=True,
                                        exclude_computed_fields=True,
                                    )
                                    if entry.rate_limits is not None
                                    else {}
                                ),
                                ensure_ascii=True,
                            ),
                            entry.source_quality,
                            json.dumps(entry.metadata, ensure_ascii=True),
                        ]
                        for entry in entries
                    ]
                    with conn.cursor() as cur:
                        cur.executemany(
                            """
                            INSERT INTO provider_models_current (
                                provider,
                                model,
                                display_name,
                                context_window,
                                max_output_tokens,
                                supports_reasoning,
                                supports_tools,
                                supports_vision,
                                pricing_json,
                                rate_limits_json,
                                source_quality,
                                metadata_json,
                                updated_at
                            ) VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s,
                                %s::jsonb, %s::jsonb, %s, %s::jsonb, now()
                            )
                            ON CONFLICT (provider, model)
                            DO UPDATE SET
                                display_name = excluded.display_name,
                                context_window = excluded.context_window,
                                max_output_tokens = excluded.max_output_tokens,
                                supports_reasoning = excluded.supports_reasoning,
                                supports_tools = excluded.supports_tools,
                                supports_vision = excluded.supports_vision,
                                pricing_json = excluded.pricing_json,
                                rate_limits_json = excluded.rate_limits_json,
                                source_quality = excluded.source_quality,
                                metadata_json = excluded.metadata_json,
                                updated_at = excluded.updated_at
                            """,
                            params,
                        )
                conn.commit()
                return len(entries)
            except Exception as exc:  # noqa: BLE001
                conn.rollback()
                raise PersistenceError(
                    f"Failed to replace provider models: {exc}"
                ) from exc

    def list_models(self, *, query: ModelCatalogQuery) -> list[ModelCatalogEntry]:
        self._runtime.init_schema()
        filter_clause, filter_params = self._model_filter_clause(query)
        with self._runtime.conn() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                rows = cur.execute(
                    SQL(
                        """
                SELECT
                    provider,
                    model,
                    display_name,
                    context_window,
                    max_output_tokens,
                    supports_reasoning,
                    supports_tools,
                    supports_vision,
                    pricing_json,
                    rate_limits_json,
                    source_quality,
                    metadata_json,
                    updated_at
                FROM provider_models_current
                {filter_clause}
                ORDER BY provider, model
                LIMIT %s OFFSET %s
                    """
                    ).format(filter_clause=filter_clause),
                    [*filter_params, int(query.limit), int(query.offset)],
                ).fetchall()
        return [_row_to_entry(row) for row in rows]

    def count_models(self, *, query: ModelCatalogQuery) -> int:
        self._runtime.init_schema()
        filter_clause, filter_params = self._model_filter_clause(query)
        with self._runtime.conn() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                row = cur.execute(
                    SQL(
                        """
                SELECT COUNT(*) AS total_count
                FROM provider_models_current
                {filter_clause}
                    """
                    ).format(filter_clause=filter_clause),
                    filter_params,
                ).fetchone()
        return int(row["total_count"]) if row is not None else 0

    def get_model(self, *, provider: str, model: str) -> ModelCatalogEntry | None:
        self._runtime.init_schema()
        with self._runtime.conn() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                row = cur.execute(
                    """
                    SELECT
                        provider,
                        model,
                        display_name,
                        context_window,
                        max_output_tokens,
                        supports_reasoning,
                        supports_tools,
                        supports_vision,
                        pricing_json,
                        rate_limits_json,
                        source_quality,
                        metadata_json,
                        updated_at
                    FROM provider_models_current
                    WHERE provider = %s AND model = %s
                    """,
                    [provider, model],
                ).fetchone()
        if row is None:
            return None
        return _row_to_entry(row)


def _row_to_entry(row: dict[str, Any]) -> ModelCatalogEntry:
    pricing = (
        ModelCatalogPricing(**row["pricing_json"])
        if isinstance(row.get("pricing_json"), dict) and row.get("pricing_json")
        else None
    )
    rate_limits = (
        ModelCatalogRateLimit(**row["rate_limits_json"])
        if isinstance(row.get("rate_limits_json"), dict) and row.get("rate_limits_json")
        else None
    )
    source_quality_raw = (
        str(row.get("source_quality"))
        if row.get("source_quality") is not None
        else "live"
    )
    if source_quality_raw == "static":
        source_quality: Literal["live", "static"] = "static"
    elif source_quality_raw == "live":
        source_quality = "live"
    else:
        raise ValueError(
            f"Unexpected source_quality in provider_models_current: {source_quality_raw!r}"
        )
    display_name_raw = row.get("display_name")
    context_window_raw = row.get("context_window")
    max_output_tokens_raw = row.get("max_output_tokens")
    supports_reasoning_raw = row.get("supports_reasoning")
    supports_tools_raw = row.get("supports_tools")
    supports_vision_raw = row.get("supports_vision")
    metadata_raw = row.get("metadata_json")

    return ModelCatalogEntry(
        provider=str(row["provider"]),
        model=str(row["model"]),
        display_name=str(display_name_raw) if display_name_raw is not None else None,
        context_window=_as_optional_int(context_window_raw),
        max_output_tokens=_as_optional_int(max_output_tokens_raw),
        supports_reasoning=(
            bool(supports_reasoning_raw) if supports_reasoning_raw is not None else None
        ),
        supports_tools=(
            bool(supports_tools_raw) if supports_tools_raw is not None else None
        ),
        supports_vision=(
            bool(supports_vision_raw) if supports_vision_raw is not None else None
        ),
        pricing=pricing,
        rate_limits=rate_limits,
        source_quality=source_quality,
        metadata=metadata_raw if isinstance(metadata_raw, dict) else {},
        fetched_at=row.get("updated_at"),
    )


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
