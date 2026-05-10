from __future__ import annotations

import json
from typing import Any, cast

import pytest
from typer.testing import CliRunner

import dr_llm.cli.models as models_cli
from dr_llm.llm.catalog.models import (
    ModelCatalogEntry,
    ModelCatalogQuery,
    ModelCatalogSyncResult,
)
from dr_llm.cli import app

runner = CliRunner()


def _fake_catalog_service(
    *,
    sync_results: list[ModelCatalogSyncResult] | None = None,
    models: list[ModelCatalogEntry] | None = None,
    count: int = 0,
) -> type:
    class _FakeService:
        def __init__(
            self, *, registry: object, repository: object = None
        ) -> None:
            _ = registry, repository

        async def sync_models_detailed(
            self, provider: str | None = None
        ) -> list[ModelCatalogSyncResult]:
            _ = provider
            return sync_results or []

        def list_models(
            self, query: ModelCatalogQuery
        ) -> list[ModelCatalogEntry]:
            _ = query
            return models or []

        def count_models(self, query: ModelCatalogQuery) -> int:
            _ = query
            return count

    return _FakeService


def _assert_blacklist_json_shape(
    payload: dict[str, object],
    *,
    provider: str,
) -> None:
    blacklist = payload.get("blacklist")
    assert isinstance(blacklist, dict)
    blacklist_dict = cast(dict[str, object], blacklist)
    assert provider in blacklist_dict
    provider_blacklist = blacklist_dict[provider]
    assert isinstance(provider_blacklist, list)
    assert provider_blacklist
    for item in provider_blacklist:
        assert isinstance(item, dict)
        item_dict = cast(dict[str, Any], item)
        assert isinstance(item_dict.get("model"), str)
        assert isinstance(item_dict.get("reason"), str)


def test_sync_verbose_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        models_cli,
        "ModelCatalogService",
        _fake_catalog_service(
            sync_results=[
                ModelCatalogSyncResult(
                    provider="openai",
                    success=True,
                    entry_count=42,
                    snapshot_id="snap_123",
                )
            ]
        ),
    )

    result = runner.invoke(
        app, ["models", "sync", "--provider", "openai", "--verbose"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload == {
        "results": [
            {
                "entry_count": 42,
                "provider": "openai",
                "raw_payload": {},
                "snapshot_id": "snap_123",
                "success": True,
            }
        ]
    }


def test_sync_failure_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        models_cli,
        "ModelCatalogService",
        _fake_catalog_service(
            sync_results=[
                ModelCatalogSyncResult(
                    provider="openai",
                    success=False,
                    error="boom\ntraceback details",
                )
            ]
        ),
    )

    result = runner.invoke(app, ["models", "sync", "--provider", "openai"])

    assert result.exit_code == 1


def test_list_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        models_cli,
        "ModelCatalogService",
        _fake_catalog_service(
            models=[
                ModelCatalogEntry(
                    provider="openai",
                    model="gpt-4.1",
                    display_name="GPT-4.1",
                )
            ]
        ),
    )

    result = runner.invoke(
        app, ["models", "list", "--provider", "openai", "--json"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["models"] == [
        {
            "display_name": "GPT-4.1",
            "metadata": {},
            "model": "gpt-4.1",
            "provider": "openai",
            "source_quality": "live",
        }
    ]
    _assert_blacklist_json_shape(payload, provider="openai")


def test_sync_list_json(monkeypatch: pytest.MonkeyPatch) -> None:
    total_count = 42
    monkeypatch.setattr(
        models_cli,
        "ModelCatalogService",
        _fake_catalog_service(
            sync_results=[
                ModelCatalogSyncResult(
                    provider="openai", success=True, entry_count=42
                )
            ],
            models=[
                ModelCatalogEntry(
                    provider="openai",
                    model="gpt-5-mini",
                    display_name="GPT-5 Mini",
                )
            ],
            count=total_count,
        ),
    )

    result = runner.invoke(
        app, ["models", "sync-list", "--provider", "openai", "--json"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["models"] == [
        {
            "display_name": "GPT-5 Mini",
            "metadata": {},
            "model": "gpt-5-mini",
            "provider": "openai",
            "source_quality": "live",
        }
    ]
    _assert_blacklist_json_shape(payload, provider="openai")


def test_list_json_includes_provider_blacklist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        models_cli,
        "ModelCatalogService",
        _fake_catalog_service(
            models=[
                ModelCatalogEntry(
                    provider="anthropic",
                    model="claude-haiku-4-5-20251001",
                    display_name="Claude Haiku 4.5",
                )
            ],
            count=1,
        ),
    )

    result = runner.invoke(
        app, ["models", "list", "--provider", "anthropic", "--json"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["models"] == [
        {
            "display_name": "Claude Haiku 4.5",
            "metadata": {},
            "model": "claude-haiku-4-5-20251001",
            "provider": "anthropic",
            "source_quality": "live",
        }
    ]
    _assert_blacklist_json_shape(payload, provider="anthropic")


def test_sync_list_failure_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        models_cli,
        "ModelCatalogService",
        _fake_catalog_service(
            sync_results=[
                ModelCatalogSyncResult(
                    provider="openai",
                    success=False,
                    error="boom\ntraceback details",
                )
            ]
        ),
    )

    result = runner.invoke(
        app, ["models", "sync-list", "--provider", "openai"]
    )

    assert result.exit_code == 1
