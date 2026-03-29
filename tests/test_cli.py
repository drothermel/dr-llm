from __future__ import annotations

import json
from typing import Any

import pytest
from typer.testing import CliRunner

import dr_llm.cli.common as cli_common
import dr_llm.cli.models as models_cli
import dr_llm.cli.query as query_cli
from dr_llm.catalog.models import ModelCatalogEntry, ModelCatalogQuery, ModelCatalogSyncResult
from dr_llm.cli import app
from dr_llm.providers.models import CallMode, Message
from dr_llm.providers.registry import ProviderRegistry
from dr_llm.providers.usage import TokenUsage
from tests.conftest import FakeAdapter, make_response

runner = CliRunner()


class _FakeRepository:
    def start_run(self, **_: object) -> str:
        return "run_cli"

    def upsert_run_parameters(
        self, *, run_id: str, parameters: dict[str, object]
    ) -> int:
        _ = run_id, parameters
        return 1

    def finish_run(self, **_: object) -> None:
        return None

    def list_calls(self, **_: object) -> list[object]:
        return []

    def close(self) -> None:
        return None


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


# --- providers ---


def test_providers_human_readable() -> None:
    result = runner.invoke(app, ["providers"])

    assert result.exit_code == 0
    assert "Providers" in result.stdout
    assert "Available" in result.stdout
    assert "Structured" in result.stdout
    assert "openai" in result.stdout
    assert "anthropic" in result.stdout
    assert "claude-code" in result.stdout
    assert '"providers"' not in result.stdout


def test_providers_json() -> None:
    result = runner.invoke(app, ["providers", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    providers = {item["provider"] for item in payload["providers"]}
    assert "openai" in providers
    assert "anthropic" in providers
    assert "google" in providers
    assert "glm" in providers
    assert "minimax" in providers
    assert "claude-code-minimax" in providers
    assert "claude-code-kimi" in providers
    for item in payload["providers"]:
        assert isinstance(item["available"], bool)
        assert isinstance(item["missing_env_vars"], list | tuple)
        assert isinstance(item["missing_executables"], list | tuple)


# --- models sync ---


def test_models_sync_concise(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_models_sync_verbose_json(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_models_sync_failure_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
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


# --- models list ---


def test_models_list_json(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_models_list_human_readable(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_models_list_without_provider_includes_provider_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_models_list_empty_page(monkeypatch: pytest.MonkeyPatch) -> None:
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


# --- query ---


def test_query_emits_response_json(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeAdapter:
        name = "openai"
        mode = "api"

        def generate(self, request: Any) -> Any:
            return make_response(
                text="hi",
                usage=TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
                provider="openai",
                model="gpt-4.1",
            )

    class _FakeRegistry:
        def get(self, name: str) -> _FakeAdapter:
            return _FakeAdapter()

    monkeypatch.setattr(query_cli, "build_default_registry", _FakeRegistry)

    result = runner.invoke(
        app,
        [
            "query",
            "--provider",
            "openai",
            "--model",
            "gpt-4.1",
            "--message",
            "hello",
            "--no-record",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["text"] == "hi"
    assert payload["usage"]["total_tokens"] == 3


# --- run start/finish ---


def test_run_start_and_finish(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_common, "_repo", lambda *_: _FakeRepository())

    start_result = runner.invoke(app, ["run", "start"])
    finish_result = runner.invoke(
        app, ["run", "finish", "--run-id", "run_cli", "--status", "success"]
    )

    assert start_result.exit_code == 0
    assert json.loads(start_result.stdout) == {
        "parameters_written": 1,
        "run_id": "run_cli",
    }
    assert finish_result.exit_code == 0
    assert json.loads(finish_result.stdout) == {
        "run_id": "run_cli",
        "status": "success",
    }


# --- catalog service (folded from test_catalog_service.py) ---


def test_catalog_service_sync_writes_snapshots(monkeypatch: pytest.MonkeyPatch) -> None:
    from dr_llm.catalog.service import ModelCatalogService
    from dr_llm.providers.provider_adapter import ProviderAdapter
    from dr_llm.providers.provider_config import ProviderConfig
    from dr_llm.providers.llm_request import LlmRequest
    from dr_llm.providers.llm_response import LlmResponse

    class _DummyAdapter(ProviderAdapter):
        def __init__(self) -> None:
            self._config = ProviderConfig(name="dummy")

        def generate(self, request: LlmRequest) -> LlmResponse:  # pragma: no cover
            return make_response(provider=request.provider, model=request.model)

    class _FakeRepo:
        def __init__(self) -> None:
            self.snapshots: list[dict[str, Any]] = []
            self.replaced: dict[str, list[ModelCatalogEntry]] = {}

        def record_model_catalog_snapshot(
            self,
            *,
            provider: str,
            status: str,
            raw_payload: dict[str, Any] | None = None,
            error_text: str | None = None,
        ) -> str:
            self.snapshots.append(
                {
                    "provider": provider,
                    "status": status,
                    "raw_payload": raw_payload or {},
                    "error_text": error_text,
                }
            )
            return f"snap-{len(self.snapshots)}"

        def replace_provider_models(
            self,
            *,
            provider: str,
            entries: list[ModelCatalogEntry],
        ) -> int:
            self.replaced[provider] = entries
            return len(entries)

        def list_models(self, *, query: Any) -> list[ModelCatalogEntry]:
            _ = query
            return []

        def count_models(self, *, query: Any) -> int:
            _ = query
            return 0

        def get_model(self, *, provider: str, model: str) -> ModelCatalogEntry | None:
            _ = (provider, model)
            return None

    registry = ProviderRegistry()
    registry.register(_DummyAdapter())
    repo = _FakeRepo()
    service = ModelCatalogService(registry=registry, repository=repo)

    def fake_fetch(adapter: Any) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
        return (
            [ModelCatalogEntry(provider=adapter.name, model="dummy-model", source_quality="live")],
            {"data": [{"id": "dummy-model"}]},
        )

    monkeypatch.setattr("dr_llm.catalog.service.fetch_models_for_adapter", fake_fetch)
    monkeypatch.setattr(
        "dr_llm.catalog.service.fetch_out_of_registry_provider_models",
        lambda provider: ([], {"data": []}),
    )

    results = service.sync_models_detailed(provider="dummy")
    assert len(results) == 1
    assert results[0].success
    assert results[0].entry_count == 1
    assert repo.snapshots
    assert repo.replaced["dummy"][0].model == "dummy-model"


# --- UI API (folded from test_ui_api.py) ---


def test_ui_api_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient
    from dr_llm.providers.provider_config import ProviderConfig
    from ui.api import main as ui_api

    adapter = FakeAdapter(
        "fake-provider",
        config=ProviderConfig(name="fake-provider", supports_structured_output=True),
    )
    registry = ProviderRegistry()
    registry.register(adapter)
    monkeypatch.setattr(ui_api, "build_default_registry", lambda: registry)

    with TestClient(ui_api.app) as client:
        response = client.get("/api/providers")

    assert response.status_code == 200
    assert response.json() == [
        {
            "provider": "fake-provider",
            "available": True,
            "missing_env_vars": [],
            "missing_executables": [],
            "supports_structured_output": True,
        }
    ]
    assert adapter.close_calls == 1
    assert getattr(ui_api.app.state, "registry", None) is None
