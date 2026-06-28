from __future__ import annotations

from datetime import datetime
import os

from dr_llm.llm import ControlMode, ProviderName
import pytest
from fastapi.testclient import TestClient

from dr_llm.llm import ProviderConfig, ProviderRegistry
from dr_llm.llm.catalog.models import ModelCatalogEntry
from dr_llm.llm.providers.impls.openrouter.orchestrator import (
    OpenRouterOrchestrator,
)
from dr_llm.llm.providers.impls.openrouter.provider import OpenRouterProvider
from dr_llm.llm.providers.transports.openai_compat.config import (
    OpenAICompatConfig,
)
from tests.conftest import FakeOrchestrator
from ui.api import main as ui_api


def test_connect_pool_database_loads_database_url_from_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "DR_LLM_DATABASE_URL=postgresql://dotenv/ui\n"
        "DR_LLM_DATABASE_URL_EXISTING=from_file\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DR_LLM_DATABASE_URL", raising=False)
    monkeypatch.delenv("DR_LLM_DATABASE_BASE_URL", raising=False)
    monkeypatch.delenv("DR_LLM_POSTGRES_SYNC_ADMIN_URL", raising=False)
    monkeypatch.setenv("DR_LLM_DATABASE_URL_EXISTING", "from_env")
    captured: dict[str, object] = {}

    def fake_connect(
        dsn: str,
        *,
        row_factory: object,
    ) -> object:
        captured["dsn"] = dsn
        captured["row_factory"] = row_factory
        return object()

    monkeypatch.setattr(ui_api.psycopg, "connect", fake_connect)

    ui_api._connect_pool_database()

    assert captured["dsn"] == "postgresql://dotenv/ui"
    assert captured["row_factory"] is ui_api.dict_row
    assert os.environ["DR_LLM_DATABASE_URL_EXISTING"] == "from_env"


def test_connect_pool_database_derives_url_from_sync_admin_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "DR_LLM_POSTGRES_SYNC_ADMIN_URL="
        "postgresql://user:pass@example.test/neondb?sslmode=require\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DR_LLM_DATABASE_URL", raising=False)
    monkeypatch.delenv("DR_LLM_DATABASE_BASE_URL", raising=False)
    monkeypatch.delenv("DR_LLM_POSTGRES_SYNC_ADMIN_URL", raising=False)
    captured: dict[str, object] = {}

    def fake_connect(
        dsn: str,
        *,
        row_factory: object,
    ) -> object:
        captured["dsn"] = dsn
        captured["row_factory"] = row_factory
        return object()

    monkeypatch.setattr(ui_api.psycopg, "connect", fake_connect)

    ui_api._connect_pool_database()

    assert captured["dsn"] == (
        "postgresql://user:pass@example.test/nl_latents?sslmode=require"
    )
    assert captured["row_factory"] is ui_api.dict_row


def test_providers_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = FakeOrchestrator(
        "fake-provider",
        config=ProviderConfig(
            name="fake-provider", supports_structured_output=True
        ),
    )
    registry = ProviderRegistry()
    registry.register(orchestrator)
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
    assert orchestrator.close_calls == 1
    assert getattr(ui_api.app.state, "registry", None) is None


def test_models_endpoint_uses_orchestrator_catalog_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = [
        ModelCatalogEntry(
            provider=ProviderName.OPENROUTER,
            model="deepseek/deepseek-chat-v3.1",
            control_mode=ControlMode.UNSUPPORTED,
            source_quality="live",
        ),
        ModelCatalogEntry(
            provider=ProviderName.OPENROUTER,
            model="deepseek/deepseek-chat",
            control_mode=ControlMode.OPENROUTER_TOGGLE,
            source_quality="live",
        ),
        ModelCatalogEntry(
            provider=ProviderName.OPENROUTER,
            model="unknown/model",
            source_quality="live",
        ),
    ]
    orchestrator = FakeOrchestrator(
        ProviderName.OPENROUTER,
        config=ProviderConfig(name=ProviderName.OPENROUTER),
        fetch_models_fn=lambda: (entries, {}),
    )
    registry = ProviderRegistry()
    registry.register(orchestrator)
    monkeypatch.setattr(ui_api, "build_default_registry", lambda: registry)

    with TestClient(ui_api.app) as client:
        response = client.get("/api/providers/openrouter/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "live"
    assert [model["model"] for model in payload["models"]] == [
        "deepseek/deepseek-chat-v3.1",
        "deepseek/deepseek-chat",
        "unknown/model",
    ]
    assert payload["models"][0]["control_mode"] == "unsupported"
    assert payload["models"][1]["control_mode"] == "openrouter_toggle"


def test_openrouter_static_models_come_from_orchestrator_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = OpenRouterOrchestrator(
        OpenRouterProvider(
            config=OpenAICompatConfig(
                name=ProviderName.OPENROUTER,
                base_url="https://openrouter.ai/api/v1",
                api_key_env="MISSING_TEST_ENV",
                required_env_vars=["MISSING_TEST_ENV"],
            )
        ),
    )
    registry = ProviderRegistry()
    registry.register(orchestrator)
    monkeypatch.setattr(ui_api, "build_default_registry", lambda: registry)

    with TestClient(ui_api.app) as client:
        response = client.get("/api/providers/openrouter/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "static"
    assert any(
        model["model"] == "openai/gpt-oss-20b" for model in payload["models"]
    )
    assert any(
        model["model"] == "deepseek/deepseek-chat-v3.1"
        for model in payload["models"]
    )


def test_nl_latents_filters_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ui_api,
        "_fetch_nl_latents_filters",
        lambda: ui_api.NlLatentsFiltersResponse(
            families=["stateful"],
            splits=["train"],
            enc_models=["openrouter:openai/gpt-5-nano"],
            budgets=["100"],
            difficulties=["3"],
            data_versions=["tasks_v2"],
        ),
    )

    with TestClient(ui_api.app) as client:
        response = client.get("/api/nl-latents/filters")

    assert response.status_code == 200
    assert response.json()["families"] == ["stateful"]
    assert response.json()["enc_models"] == ["openrouter:openai/gpt-5-nano"]


def test_nl_latents_samples_endpoint_forwards_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_fetch(**kwargs: object) -> ui_api.NlLatentsSamplesResponse:
        captured.update(kwargs)
        return ui_api.NlLatentsSamplesResponse(
            samples=[
                ui_api.NlLatentsSampleListRow(
                    sample_id="sample-1",
                    family="stateful",
                    difficulty="3",
                    split="train",
                    language="python",
                    budget="100",
                    enc_model="openrouter:openai/gpt-5-nano",
                    dec_model="openrouter:openai/gpt-5-nano",
                    enc_model_label="openai/gpt-5-nano",
                    dec_model_label="openai/gpt-5-nano",
                    status="active",
                    attempt_count=1,
                    created_at=datetime(2026, 2, 16, 22, 26),
                    result_state="passed",
                    prompt_config_label="Noop / High Level",
                )
            ],
            total=1,
            page=2,
            limit=10,
            total_pages=1,
        )

    monkeypatch.setattr(ui_api, "_fetch_nl_latents_samples", fake_fetch)

    with TestClient(ui_api.app) as client:
        response = client.get(
            "/api/nl-latents/samples",
            params={
                "page": 2,
                "limit": 10,
                "family": "stateful",
                "hide_pending": "true",
            },
        )

    assert response.status_code == 200
    assert captured["page"] == 2
    assert captured["limit"] == 10
    assert captured["family"] == "stateful"
    assert captured["hide_pending"] is True
    payload = response.json()
    assert payload["samples"][0]["sample_id"] == "sample-1"
    assert payload["samples"][0]["prompt_config_label"] == "Noop / High Level"


def test_nl_latents_conditions_parameterizes_smoke_filter() -> None:
    where, values = ui_api._nl_latents_conditions(
        family=None,
        split=None,
        enc_model=None,
        budget=None,
        difficulty=None,
        data_version=None,
        result=None,
        hide_pending=False,
        hide_smoke=True,
    )

    assert "coalesce(run_id, '') NOT ILIKE %s" in where
    assert "%smoke%" in values


def test_nl_latents_sample_detail_endpoint_returns_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ui_api, "_fetch_nl_latents_sample", lambda _id: None)

    with TestClient(ui_api.app) as client:
        response = client.get("/api/nl-latents/samples/missing")

    assert response.status_code == 404


def test_published_filters_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ui_api,
        "_fetch_published_filters",
        lambda: ui_api.PublishedFiltersResponse(
            projects=["code_comp_v0"],
            source_pools=["direct_enc_t0"],
            sample_roles=["encoder_description"],
            task_families=["stateful"],
            models=["openai/gpt-5-nano"],
            result_states=["completed"],
            datasets=["stateful/1"],
        ),
    )

    with TestClient(ui_api.app) as client:
        response = client.get("/api/published/filters")

    assert response.status_code == 200
    payload = response.json()
    assert payload["projects"] == ["code_comp_v0"]
    assert payload["sample_roles"] == ["encoder_description"]


def test_published_samples_endpoint_forwards_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_fetch(**kwargs: object) -> ui_api.PublishedSamplesResponse:
        captured.update(kwargs)
        return ui_api.PublishedSamplesResponse(
            samples=[
                ui_api.PublishedSampleListRow(
                    source_project="code_comp_v0",
                    source_pool="direct_enc_t0",
                    source_sample_id="sample-1",
                    sample_idx=1,
                    created_at=datetime(2026, 2, 16, 22, 26),
                    sample_role="encoder_description",
                    output_kind="text",
                    task_family="stateful",
                    model="openai/gpt-5-nano",
                    result_state="completed",
                )
            ],
            total=1,
            page=2,
            limit=10,
            total_pages=1,
        )

    monkeypatch.setattr(ui_api, "_fetch_published_samples", fake_fetch)

    with TestClient(ui_api.app) as client:
        response = client.get(
            "/api/published/samples",
            params={
                "page": 2,
                "limit": 10,
                "project": "code_comp_v0",
                "source_pool": "direct_enc_t0",
                "hide_smoke": "true",
            },
        )

    assert response.status_code == 200
    assert captured["page"] == 2
    assert captured["limit"] == 10
    assert captured["project"] == "code_comp_v0"
    assert captured["source_pool"] == "direct_enc_t0"
    assert captured["hide_smoke"] is True
    assert response.json()["samples"][0]["source_sample_id"] == "sample-1"


def test_published_conditions_parameterizes_smoke_filter() -> None:
    where, values = ui_api._published_conditions(
        project=None,
        source_pool=None,
        sample_role=None,
        task_family=None,
        model=None,
        result=None,
        dataset=None,
        hide_pending=False,
        hide_smoke=True,
    )

    assert "coalesce(run_id, '') NOT ILIKE %s" in where
    assert "coalesce(source_pool, '') NOT ILIKE %s" in where
    assert values == ["%smoke%", "%smoke%"]


def test_published_sample_detail_endpoint_returns_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ui_api,
        "_fetch_published_sample",
        lambda _project, _pool, _id: None,
    )

    with TestClient(ui_api.app) as client:
        response = client.get(
            "/api/published/samples/code_comp_v0/direct_enc_t0/missing"
        )

    assert response.status_code == 404
