from __future__ import annotations

import json

from typer.testing import CliRunner

import dr_llm.cli.common as cli_common
import dr_llm.cli.models as models_cli
import dr_llm.cli.query as query_cli
from dr_llm.catalog.models import (
    ModelCatalogEntry,
    ModelCatalogQuery,
    ModelCatalogSyncResult,
)
from dr_llm.cli import app
from dr_llm.generation.models import (
    CallMode,
    LlmRequest,
    LlmResponse,
    Message,
    TokenUsage,
)


class _CliFakeRepository:
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


def test_providers_command_is_human_readable_by_default() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["providers"])

    assert result.exit_code == 0
    assert "Providers" in result.stdout
    assert "Available" in result.stdout
    assert "Structured" in result.stdout
    assert "openai" in result.stdout
    assert "anthropic" in result.stdout
    assert "claude-code" in result.stdout
    assert '"providers"' not in result.stdout


def test_providers_command_json_lists_known_providers() -> None:
    runner = CliRunner()
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


def test_models_sync_is_concise_by_default(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr(cli_common, "_repo", lambda *_: _CliFakeRepository())

    class _FakeClient:
        def __init__(self, *, registry: object, repository: object) -> None:
            _ = registry, repository

        def sync_models_detailed(
            self, provider: str | None = None
        ) -> list[ModelCatalogSyncResult]:
            assert provider == "openai"
            return [
                ModelCatalogSyncResult(
                    provider="openai",
                    success=True,
                    entry_count=42,
                )
            ]

    monkeypatch.setattr(models_cli, "LlmClient", _FakeClient)

    result = runner.invoke(app, ["models", "sync", "--provider", "openai"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "Synced 42 models for openai."


def test_models_sync_verbose_emits_json(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr(cli_common, "_repo", lambda *_: _CliFakeRepository())

    class _FakeClient:
        def __init__(self, *, registry: object, repository: object) -> None:
            _ = registry, repository

        def sync_models_detailed(
            self, provider: str | None = None
        ) -> list[ModelCatalogSyncResult]:
            assert provider == "openai"
            return [
                ModelCatalogSyncResult(
                    provider="openai",
                    success=True,
                    entry_count=42,
                    snapshot_id="snap_123",
                )
            ]

    monkeypatch.setattr(models_cli, "LlmClient", _FakeClient)

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


def test_models_sync_failure_is_concise_and_nonzero(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr(cli_common, "_repo", lambda *_: _CliFakeRepository())

    class _FakeClient:
        def __init__(self, *, registry: object, repository: object) -> None:
            _ = registry, repository

        def sync_models_detailed(
            self, provider: str | None = None
        ) -> list[ModelCatalogSyncResult]:
            assert provider == "openai"
            return [
                ModelCatalogSyncResult(
                    provider="openai",
                    success=False,
                    error="boom\ntraceback details",
                )
            ]

    monkeypatch.setattr(models_cli, "LlmClient", _FakeClient)

    result = runner.invoke(app, ["models", "sync", "--provider", "openai"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr.strip() == "Model sync failed for openai: boom"


def test_models_list_json_emits_models(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr(cli_common, "_repo", lambda *_: _CliFakeRepository())

    class _FakeClient:
        def __init__(self, *, registry: object, repository: object) -> None:
            _ = registry, repository

        def list_models(self, query: ModelCatalogQuery) -> list[ModelCatalogEntry]:
            assert query.provider == "openai"
            return [
                ModelCatalogEntry(
                    provider="openai",
                    model="gpt-4.1",
                    display_name="GPT-4.1",
                )
            ]

    monkeypatch.setattr(models_cli, "LlmClient", _FakeClient)

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


def test_models_list_is_human_readable_with_provider_header(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr(cli_common, "_repo", lambda *_: _CliFakeRepository())

    class _FakeClient:
        def __init__(self, *, registry: object, repository: object) -> None:
            _ = registry, repository

        def list_models(self, query: ModelCatalogQuery) -> list[ModelCatalogEntry]:
            assert query.provider == "openai"
            assert query.limit == 20
            return [
                ModelCatalogEntry(
                    provider="openai",
                    model="gpt-4o-mini",
                    display_name="GPT-4o mini",
                ),
                ModelCatalogEntry(
                    provider="openai",
                    model="gpt-4.1",
                ),
            ]

        def count_models(self, query: ModelCatalogQuery) -> int:
            assert query.provider == "openai"
            return 347

    monkeypatch.setattr(models_cli, "LlmClient", _FakeClient)

    result = runner.invoke(app, ["models", "list", "--provider", "openai"])

    assert result.exit_code == 0
    assert result.stdout == (
        "openai Models (Showing 2 out of 347)\n- gpt-4o-mini (GPT-4o mini)\n- gpt-4.1\n"
    )


def test_models_list_without_provider_includes_provider_prefix(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr(cli_common, "_repo", lambda *_: _CliFakeRepository())

    class _FakeClient:
        def __init__(self, *, registry: object, repository: object) -> None:
            _ = registry, repository

        def list_models(self, query: ModelCatalogQuery) -> list[ModelCatalogEntry]:
            assert query.provider is None
            assert query.limit == 20
            return [
                ModelCatalogEntry(
                    provider="anthropic",
                    model="claude-sonnet-4",
                ),
                ModelCatalogEntry(
                    provider="openai",
                    model="gpt-4o-mini",
                ),
            ]

        def count_models(self, query: ModelCatalogQuery) -> int:
            assert query.provider is None
            return 347

    monkeypatch.setattr(models_cli, "LlmClient", _FakeClient)

    result = runner.invoke(app, ["models", "list"])

    assert result.exit_code == 0
    assert result.stdout == (
        "Models (Showing 2 out of 347 across 2 providers)\n"
        "- anthropic: claude-sonnet-4\n"
        "- openai: gpt-4o-mini\n"
    )


def test_models_list_empty_page_mentions_matching_total(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr(cli_common, "_repo", lambda *_: _CliFakeRepository())

    class _FakeClient:
        def __init__(self, *, registry: object, repository: object) -> None:
            _ = registry, repository

        def list_models(self, query: ModelCatalogQuery) -> list[ModelCatalogEntry]:
            assert query.provider == "openai"
            assert query.offset == 40
            return []

        def count_models(self, query: ModelCatalogQuery) -> int:
            assert query.provider == "openai"
            return 347

    monkeypatch.setattr(models_cli, "LlmClient", _FakeClient)

    result = runner.invoke(
        app, ["models", "list", "--provider", "openai", "--offset", "40"]
    )

    assert result.exit_code == 0
    assert (
        result.stdout.strip()
        == "No models found on this page for openai. 347 matching models exist."
    )


def test_query_emits_response_json(monkeypatch) -> None:
    runner = CliRunner()

    class _FakeClient:
        def __init__(self, *, registry: object, repository: object | None) -> None:
            _ = registry, repository

        def query(self, request: LlmRequest, **_: object) -> LlmResponse:
            assert request.provider == "openai"
            assert request.model == "gpt-4.1"
            assert request.messages == [Message(role="user", content="hello")]
            return LlmResponse(
                text="hi",
                usage=TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
                provider="openai",
                model="gpt-4.1",
                mode=CallMode.api,
            )

    monkeypatch.setattr(query_cli, "LlmClient", _FakeClient)

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


def test_run_start_and_finish_emit_json(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr(cli_common, "_repo", lambda *_: _CliFakeRepository())

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
