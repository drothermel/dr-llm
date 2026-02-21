from __future__ import annotations

import json
from typing import Any, Literal
from uuid import uuid4

from llm_pool.errors import PersistenceError
from llm_pool.storage._runtime import StorageRuntime
from llm_pool.types import (
    ModelCatalogEntry,
    ModelCatalogPricing,
    ModelCatalogQuery,
    ModelCatalogRateLimit,
)


class CatalogStore:
    def __init__(self, runtime: StorageRuntime) -> None:
        self._runtime = runtime

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
                conn.commit()
                return snapshot_id
            except Exception as exc:  # noqa: BLE001
                conn.rollback()
                raise PersistenceError(
                    f"Failed to record model catalog snapshot: {exc}"
                ) from exc

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
                count = 0
                for entry in entries:
                    conn.execute(
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
                        ],
                    )
                    count += 1
                conn.commit()
                return count
            except Exception as exc:  # noqa: BLE001
                conn.rollback()
                raise PersistenceError(
                    f"Failed to replace provider models: {exc}"
                ) from exc

    def upsert_model_overrides(
        self,
        *,
        entries: list[ModelCatalogEntry],
    ) -> int:
        self._runtime.init_schema()
        with self._runtime.conn() as conn:
            count = 0
            try:
                for entry in entries:
                    if entry.pricing is None and entry.rate_limits is None:
                        continue
                    conn.execute(
                        """
                        INSERT INTO provider_model_overrides (
                            provider,
                            model,
                            pricing_json,
                            rate_limits_json,
                            notes,
                            updated_at
                        ) VALUES (%s, %s, %s::jsonb, %s::jsonb, %s, now())
                        ON CONFLICT (provider, model)
                        DO UPDATE SET
                            pricing_json = excluded.pricing_json,
                            rate_limits_json = excluded.rate_limits_json,
                            notes = excluded.notes,
                            updated_at = excluded.updated_at
                        """,
                        [
                            entry.provider,
                            entry.model,
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
                            str(entry.metadata.get("notes") or ""),
                        ],
                    )
                    count += 1
                conn.commit()
                return count
            except Exception as exc:  # noqa: BLE001
                conn.rollback()
                raise PersistenceError(
                    f"Failed to upsert model overrides: {exc}"
                ) from exc

    def list_models(self, *, query: ModelCatalogQuery) -> list[ModelCatalogEntry]:
        self._runtime.init_schema()
        with self._runtime.conn() as conn:
            rows = conn.execute(
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
            WHERE (%s::text IS NULL OR provider = %s)
              AND (%s::boolean IS NULL OR supports_reasoning = %s)
              AND (%s::text IS NULL OR model ILIKE %s)
            ORDER BY provider, model
            LIMIT %s OFFSET %s
                """,
                [
                    query.provider,
                    query.provider,
                    query.supports_reasoning,
                    query.supports_reasoning,
                    query.model_contains,
                    (
                        f"%{query.model_contains}%"
                        if query.model_contains is not None
                        else None
                    ),
                    int(query.limit),
                    int(query.offset),
                ],
            ).fetchall()
        return [_row_to_entry(row) for row in rows]

    def get_model(self, *, provider: str, model: str) -> ModelCatalogEntry | None:
        self._runtime.init_schema()
        with self._runtime.conn() as conn:
            row = conn.execute(
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


def _row_to_entry(row: tuple[Any, ...]) -> ModelCatalogEntry:
    pricing = (
        ModelCatalogPricing(**row[8]) if isinstance(row[8], dict) and row[8] else None
    )
    rate_limits = (
        ModelCatalogRateLimit(**row[9]) if isinstance(row[9], dict) and row[9] else None
    )
    source_quality_raw = str(row[10]) if row[10] is not None else "live"
    if source_quality_raw == "overlay":
        source_quality: Literal["live", "overlay", "static"] = "overlay"
    elif source_quality_raw == "static":
        source_quality = "static"
    else:
        source_quality = "live"
    return ModelCatalogEntry(
        provider=str(row[0]),
        model=str(row[1]),
        display_name=str(row[2]) if row[2] is not None else None,
        context_window=int(row[3]) if row[3] is not None else None,
        max_output_tokens=int(row[4]) if row[4] is not None else None,
        supports_reasoning=bool(row[5]) if row[5] is not None else None,
        supports_tools=bool(row[6]) if row[6] is not None else None,
        supports_vision=bool(row[7]) if row[7] is not None else None,
        pricing=pricing,
        rate_limits=rate_limits,
        source_quality=source_quality,
        metadata=row[11] if isinstance(row[11], dict) else {},
        fetched_at=row[12],
    )
