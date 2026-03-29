from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

import dr_llm.cli.models as models_cli
from dr_llm.catalog.models import ModelCatalogEntry, ModelCatalogQuery, ModelCatalogSyncResult
from dr_llm.cli import app

runner = CliRunner()


def _fake_catalog_service(
    *,
    sync_results: list[ModelCatalogSyncResult] | None = None,
    models: list[ModelCatalogEntry] | None = None,
    count: int = 0,
) -> type:
    class _FakeService:
        def __init__(self, *, registry: object, repository: object = None) -> None:
            _ = registry, repository

        def sync_models_detailed(
            self, provider: str | None = None
        ) -> list[ModelCatalogSyncResult]:
            _ = provider
            return sync_results or []

        def list_models(self, query: ModelCatalogQuery) -> list[ModelCatalogEntry]:
            _ = query
            return models or []

        def count_models(self, query: ModelCatalogQuery) -> int:
            _ = query
            return count

    return _FakeService


def test_sync_concise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        models_cli,
        "ModelCatalogService",
        _fake_catalog_service(
            sync_results=[
                ModelCatalogSyncResult(provider="openai", success=True, entry_count=42)
            ]
        ),
    )

    result = runner.invoke(app, ["models", "sync", "--provider", "openai"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "Synced 42 models for openai."


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

    result = runner.invoke(app, ["models", "sync", "--provider", "openai", "--verbose"])

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
    assert result.stdout == ""
    assert result.stderr.strip() == "Model sync failed for openai: boom"


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

    result = runner.invoke(app, ["models", "list", "--provider", "openai", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload == {
        "models": [
            {
                "display_name": "GPT-4.1",
                "metadata": {},
                "model": "gpt-4.1",
                "provider": "openai",
                "source_quality": "live",
            }
        ]
    }


def test_list_human_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        models_cli,
        "ModelCatalogService",
        _fake_catalog_service(
            models=[
                ModelCatalogEntry(
                    provider="openai",
                    model="gpt-4o-mini",
                    display_name="GPT-4o mini",
                ),
                ModelCatalogEntry(provider="openai", model="gpt-4.1"),
            ],
            count=347,
        ),
    )

    result = runner.invoke(app, ["models", "list", "--provider", "openai"])

    assert result.exit_code == 0
    assert result.stdout == (
        "openai Models (Showing 2 out of 347)\n- gpt-4o-mini (GPT-4o mini)\n- gpt-4.1\n"
    )


def test_list_multi_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        models_cli,
        "ModelCatalogService",
        _fake_catalog_service(
            models=[
                ModelCatalogEntry(provider="anthropic", model="claude-sonnet-4"),
                ModelCatalogEntry(provider="openai", model="gpt-4o-mini"),
            ],
            count=347,
        ),
    )

    result = runner.invoke(app, ["models", "list"])

    assert result.exit_code == 0
    assert result.stdout == (
        "Models (Showing 2 out of 347 across 2 providers)\n"
        "- anthropic: claude-sonnet-4\n"
        "- openai: gpt-4o-mini\n"
    )


def test_list_empty_page(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        models_cli,
        "ModelCatalogService",
        _fake_catalog_service(models=[], count=347),
    )

    result = runner.invoke(
        app, ["models", "list", "--provider", "openai", "--offset", "40"]
    )

    assert result.exit_code == 0
    assert (
        result.stdout.strip()
        == "No models found on this page for openai. 347 matching models exist."
    )
