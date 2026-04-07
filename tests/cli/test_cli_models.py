from __future__ import annotations

import json
from typing import Any, cast

import pytest
from typer.testing import CliRunner

import dr_llm.cli.models as models_cli
from dr_llm.catalog.model_blacklist import blacklisted_models
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


def _blacklist_json(provider: str | None = None) -> dict[str, list[dict[str, str]]]:
    return {
        provider_name: [
            {
                "model": item.model,
                "reason": item.reason,
            }
            for item in items
        ]
        for provider_name, items in blacklisted_models(provider=provider).items()
    }


def _blacklist_text(provider: str | None = None) -> str:
    grouped = blacklisted_models(provider=provider)
    if not grouped:
        return ""

    lines = [""]
    if provider is not None:
        lines.append(f"Blacklisted Models for {provider}")
        lines.extend(
            f"- {item.model}: {item.reason}" for item in grouped.get(provider, [])
        )
        return "\n".join(lines) + "\n"

    lines.append("Blacklisted Models")
    for provider_name, items in grouped.items():
        lines.append(f"{provider_name}:")
        lines.extend(f"- {item.model}: {item.reason}" for item in items)
    return "\n".join(lines) + "\n"


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


def _assert_blacklist_text_present(stdout: str, *, provider: str) -> None:
    grouped = blacklisted_models(provider=provider)
    assert grouped
    assert f"Blacklisted Models for {provider}" in stdout
    sample = grouped[provider][0]
    assert f"- {sample.model}: {sample.reason}" in stdout


def _assert_grouped_blacklist_text_present(stdout: str) -> None:
    grouped = blacklisted_models()
    assert grouped
    assert "Blacklisted Models" in stdout
    for provider_name, items in grouped.items():
        assert f"{provider_name}:" in stdout
        if items:
            sample = items[0]
            assert f"- {sample.model}: {sample.reason}" in stdout


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


def test_list_human_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    total_count = 347
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
            count=total_count,
        ),
    )

    result = runner.invoke(app, ["models", "list", "--provider", "openai"])

    assert result.exit_code == 0
    assert f"openai Models (Showing 2 out of {total_count})" in result.stdout
    assert "- gpt-4o-mini (GPT-4o mini)" in result.stdout
    assert "- gpt-4.1" in result.stdout
    _assert_blacklist_text_present(result.stdout, provider="openai")


def test_list_multi_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    total_count = 347
    monkeypatch.setattr(
        models_cli,
        "ModelCatalogService",
        _fake_catalog_service(
            models=[
                ModelCatalogEntry(provider="anthropic", model="claude-sonnet-4"),
                ModelCatalogEntry(provider="openai", model="gpt-4o-mini"),
            ],
            count=total_count,
        ),
    )

    result = runner.invoke(app, ["models", "list"])

    assert result.exit_code == 0
    assert f"Models (Showing 2 out of {total_count} across 2 providers)" in result.stdout
    assert "- anthropic: claude-sonnet-4" in result.stdout
    assert "- openai: gpt-4o-mini" in result.stdout
    _assert_grouped_blacklist_text_present(result.stdout)


def test_list_empty_page(monkeypatch: pytest.MonkeyPatch) -> None:
    total_count = 347
    monkeypatch.setattr(
        models_cli,
        "ModelCatalogService",
        _fake_catalog_service(models=[], count=total_count),
    )

    result = runner.invoke(
        app, ["models", "list", "--provider", "openai", "--offset", "40"]
    )

    assert result.exit_code == 0
    assert result.stdout.startswith(
        f"No models found on this page for openai. {total_count} matching models exist.\n"
    )
    _assert_blacklist_text_present(result.stdout, provider="openai")


def test_sync_list_runs_sync_then_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    total_count = 42
    monkeypatch.setattr(
        models_cli,
        "ModelCatalogService",
        _fake_catalog_service(
            sync_results=[
                ModelCatalogSyncResult(provider="openai", success=True, entry_count=42)
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

    result = runner.invoke(app, ["models", "sync-list", "--provider", "openai"])

    assert result.exit_code == 0
    assert f"openai Models (Showing 1 out of {total_count})" in result.stdout
    assert "- gpt-5-mini (GPT-5 Mini)" in result.stdout
    _assert_blacklist_text_present(result.stdout, provider="openai")


def test_sync_list_json(monkeypatch: pytest.MonkeyPatch) -> None:
    total_count = 42
    monkeypatch.setattr(
        models_cli,
        "ModelCatalogService",
        _fake_catalog_service(
            sync_results=[
                ModelCatalogSyncResult(provider="openai", success=True, entry_count=42)
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


def test_list_human_readable_shows_provider_blacklist(
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

    result = runner.invoke(app, ["models", "list", "--provider", "anthropic"])

    assert result.exit_code == 0
    assert "anthropic Models (Showing 1 out of 1)" in result.stdout
    assert "- claude-haiku-4-5-20251001 (Claude Haiku 4.5)" in result.stdout
    _assert_blacklist_text_present(result.stdout, provider="anthropic")


def test_list_json_includes_provider_blacklist(monkeypatch: pytest.MonkeyPatch) -> None:
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

    result = runner.invoke(app, ["models", "list", "--provider", "anthropic", "--json"])

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


def test_sync_list_failure_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
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

    result = runner.invoke(app, ["models", "sync-list", "--provider", "openai"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr.strip() == "Model sync failed for openai: boom"
