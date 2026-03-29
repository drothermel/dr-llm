from __future__ import annotations

from pathlib import Path

import pytest

from dr_llm.catalog.file_store import FileCatalogStore
from dr_llm.catalog.models import ModelCatalogEntry, ModelCatalogQuery


@pytest.fixture
def store(tmp_path: Path) -> FileCatalogStore:
    return FileCatalogStore(cache_dir=tmp_path)


def _entry(provider: str, model: str, **kwargs: object) -> ModelCatalogEntry:
    return ModelCatalogEntry(provider=provider, model=model, **kwargs)  # type: ignore[arg-type]


def test_round_trip(store: FileCatalogStore) -> None:
    entries = [
        _entry("openai", "gpt-4.1", display_name="GPT-4.1"),
        _entry("openai", "gpt-4o-mini"),
    ]
    count = store.replace_provider_models(provider="openai", entries=entries)
    assert count == 2

    result = store.list_models(query=ModelCatalogQuery(provider="openai"))
    assert len(result) == 2
    assert result[0].model == "gpt-4.1"
    assert result[0].display_name == "GPT-4.1"


def test_replace_overwrites(store: FileCatalogStore) -> None:
    store.replace_provider_models(
        provider="openai", entries=[_entry("openai", "old-model")]
    )
    store.replace_provider_models(
        provider="openai", entries=[_entry("openai", "new-model")]
    )
    result = store.list_models(query=ModelCatalogQuery(provider="openai"))
    assert len(result) == 1
    assert result[0].model == "new-model"


def test_filter_by_provider(store: FileCatalogStore) -> None:
    store.replace_provider_models(
        provider="openai", entries=[_entry("openai", "gpt-4.1")]
    )
    store.replace_provider_models(
        provider="anthropic", entries=[_entry("anthropic", "claude-sonnet-4")]
    )

    openai = store.list_models(query=ModelCatalogQuery(provider="openai"))
    assert len(openai) == 1
    assert openai[0].provider == "openai"

    all_models = store.list_models(query=ModelCatalogQuery())
    assert len(all_models) == 2


def test_filter_by_supports_reasoning(store: FileCatalogStore) -> None:
    store.replace_provider_models(
        provider="test",
        entries=[
            _entry("test", "reasoner", supports_reasoning=True),
            _entry("test", "basic", supports_reasoning=False),
        ],
    )
    result = store.list_models(
        query=ModelCatalogQuery(provider="test", supports_reasoning=True)
    )
    assert len(result) == 1
    assert result[0].model == "reasoner"


def test_filter_by_model_contains(store: FileCatalogStore) -> None:
    store.replace_provider_models(
        provider="openai",
        entries=[
            _entry("openai", "gpt-4.1"),
            _entry("openai", "gpt-4o-mini"),
            _entry("openai", "o3-pro"),
        ],
    )
    result = store.list_models(
        query=ModelCatalogQuery(provider="openai", model_contains="gpt")
    )
    assert len(result) == 2


def test_model_contains_is_case_insensitive(store: FileCatalogStore) -> None:
    store.replace_provider_models(
        provider="openai", entries=[_entry("openai", "GPT-4.1")]
    )
    result = store.list_models(
        query=ModelCatalogQuery(provider="openai", model_contains="gpt")
    )
    assert len(result) == 1


def test_get_model_hit(store: FileCatalogStore) -> None:
    store.replace_provider_models(
        provider="openai", entries=[_entry("openai", "gpt-4.1")]
    )
    result = store.get_model(provider="openai", model="gpt-4.1")
    assert result is not None
    assert result.model == "gpt-4.1"


def test_get_model_miss(store: FileCatalogStore) -> None:
    store.replace_provider_models(
        provider="openai", entries=[_entry("openai", "gpt-4.1")]
    )
    assert store.get_model(provider="openai", model="nonexistent") is None
    assert store.get_model(provider="nonexistent", model="gpt-4.1") is None


def test_count_models(store: FileCatalogStore) -> None:
    store.replace_provider_models(
        provider="openai",
        entries=[_entry("openai", "a"), _entry("openai", "b"), _entry("openai", "c")],
    )
    assert store.count_models(query=ModelCatalogQuery(provider="openai")) == 3
    assert store.count_models(query=ModelCatalogQuery(provider="nonexistent")) == 0


def test_limit_and_offset(store: FileCatalogStore) -> None:
    entries = [_entry("p", f"model-{i}") for i in range(10)]
    store.replace_provider_models(provider="p", entries=entries)

    page = store.list_models(query=ModelCatalogQuery(provider="p", limit=3, offset=2))
    assert len(page) == 3
    assert page[0].model == "model-2"


def test_snapshot_returns_id(store: FileCatalogStore) -> None:
    sid = store.record_model_catalog_snapshot(provider="openai", status="success")
    assert isinstance(sid, str)
    assert len(sid) > 0


def test_empty_store_returns_empty(store: FileCatalogStore) -> None:
    assert store.list_models(query=ModelCatalogQuery()) == []
    assert store.count_models(query=ModelCatalogQuery()) == 0
    assert store.get_model(provider="x", model="y") is None


def test_file_exists_after_replace(store: FileCatalogStore, tmp_path: Path) -> None:
    store.replace_provider_models(
        provider="openai", entries=[_entry("openai", "gpt-4.1")]
    )
    assert (tmp_path / "openai.json").exists()
