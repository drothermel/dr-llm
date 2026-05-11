from __future__ import annotations

from dr_llm.llm import ControlMode, ProviderName
from pathlib import Path
from typing import Any

import pytest

from dr_llm.llm.catalog.file_store import (
    CatalogCacheCorruptError,
    FileCatalogStore,
)
from dr_llm.llm.catalog.models import ModelCatalogEntry, ModelCatalogQuery


@pytest.fixture
def store(tmp_path: Path) -> FileCatalogStore:
    return FileCatalogStore(cache_dir=tmp_path)


def _entry(provider: str, model: str, **kwargs: Any) -> ModelCatalogEntry:
    return ModelCatalogEntry(provider=provider, model=model, **kwargs)


def test_round_trip(store: FileCatalogStore) -> None:
    entries = [
        _entry(ProviderName.OPENAI, "gpt-4.1", display_name="GPT-4.1"),
        _entry(ProviderName.OPENAI, "gpt-4o-mini"),
    ]
    count = store.replace_provider_models(
        provider=ProviderName.OPENAI, entries=entries
    )
    assert count == 2

    result = store.list_models(
        query=ModelCatalogQuery(provider=ProviderName.OPENAI)
    )
    assert len(result) == 2
    assert result[0].model == "gpt-4.1"
    assert result[0].display_name == "GPT-4.1"


def test_replace_overwrites(store: FileCatalogStore) -> None:
    store.replace_provider_models(
        provider=ProviderName.OPENAI,
        entries=[_entry(ProviderName.OPENAI, "old-model")],
    )
    store.replace_provider_models(
        provider=ProviderName.OPENAI,
        entries=[_entry(ProviderName.OPENAI, "new-model")],
    )
    result = store.list_models(
        query=ModelCatalogQuery(provider=ProviderName.OPENAI)
    )
    assert len(result) == 1
    assert result[0].model == "new-model"


def test_filter_by_provider(store: FileCatalogStore) -> None:
    store.replace_provider_models(
        provider=ProviderName.OPENAI,
        entries=[_entry(ProviderName.OPENAI, "gpt-4.1")],
    )
    store.replace_provider_models(
        provider=ProviderName.ANTHROPIC,
        entries=[_entry(ProviderName.ANTHROPIC, "claude-sonnet-4")],
    )

    openai = store.list_models(
        query=ModelCatalogQuery(provider=ProviderName.OPENAI)
    )
    assert len(openai) == 1
    assert openai[0].provider == ProviderName.OPENAI

    all_models = store.list_models(query=ModelCatalogQuery())
    assert len(all_models) == 2


def test_filter_by_control_mode(store: FileCatalogStore) -> None:
    store.replace_provider_models(
        provider="test",
        entries=[
            _entry(
                "test",
                "reasoner",
                control_mode=ControlMode.OPENAI_EFFORT,
            ),
            _entry("test", "basic", control_mode=ControlMode.UNSUPPORTED),
        ],
    )
    result = store.list_models(
        query=ModelCatalogQuery(
            provider="test", control_mode=ControlMode.OPENAI_EFFORT
        )
    )
    assert len(result) == 1
    assert result[0].model == "reasoner"


def test_filter_by_model_contains(store: FileCatalogStore) -> None:
    store.replace_provider_models(
        provider=ProviderName.OPENAI,
        entries=[
            _entry(ProviderName.OPENAI, "gpt-4.1"),
            _entry(ProviderName.OPENAI, "gpt-4o-mini"),
            _entry(ProviderName.OPENAI, "o3-pro"),
        ],
    )
    result = store.list_models(
        query=ModelCatalogQuery(
            provider=ProviderName.OPENAI, model_contains="gpt"
        )
    )
    assert len(result) == 2


def test_model_contains_is_case_insensitive(store: FileCatalogStore) -> None:
    store.replace_provider_models(
        provider=ProviderName.OPENAI,
        entries=[_entry(ProviderName.OPENAI, "GPT-4.1")],
    )
    result = store.list_models(
        query=ModelCatalogQuery(
            provider=ProviderName.OPENAI, model_contains="gpt"
        )
    )
    assert len(result) == 1


def test_get_model_hit(store: FileCatalogStore) -> None:
    store.replace_provider_models(
        provider=ProviderName.OPENAI,
        entries=[_entry(ProviderName.OPENAI, "gpt-4.1")],
    )
    result = store.get_model(provider=ProviderName.OPENAI, model="gpt-4.1")
    assert result is not None
    assert result.model == "gpt-4.1"


def test_get_model_miss(store: FileCatalogStore) -> None:
    store.replace_provider_models(
        provider=ProviderName.OPENAI,
        entries=[_entry(ProviderName.OPENAI, "gpt-4.1")],
    )
    assert (
        store.get_model(provider=ProviderName.OPENAI, model="nonexistent")
        is None
    )
    assert store.get_model(provider="nonexistent", model="gpt-4.1") is None


def test_count_models(store: FileCatalogStore) -> None:
    store.replace_provider_models(
        provider=ProviderName.OPENAI,
        entries=[
            _entry(ProviderName.OPENAI, "a"),
            _entry(ProviderName.OPENAI, "b"),
            _entry(ProviderName.OPENAI, "c"),
        ],
    )
    assert (
        store.count_models(
            query=ModelCatalogQuery(provider=ProviderName.OPENAI)
        )
        == 3
    )
    assert (
        store.count_models(query=ModelCatalogQuery(provider="nonexistent"))
        == 0
    )


def test_limit_and_offset(store: FileCatalogStore) -> None:
    entries = [_entry("p", f"model-{i}") for i in range(10)]
    store.replace_provider_models(provider="p", entries=entries)

    page = store.list_models(
        query=ModelCatalogQuery(provider="p", limit=3, offset=2)
    )
    assert len(page) == 3
    assert page[0].model == "model-2"


def test_snapshot_returns_id(store: FileCatalogStore) -> None:
    sid = store.record_model_catalog_snapshot(
        provider=ProviderName.OPENAI, status="success"
    )
    assert isinstance(sid, str)
    assert len(sid) > 0


def test_empty_store_returns_empty(store: FileCatalogStore) -> None:
    assert store.list_models(query=ModelCatalogQuery()) == []
    assert store.count_models(query=ModelCatalogQuery()) == 0
    assert store.get_model(provider="x", model="y") is None


def test_file_exists_after_replace(
    store: FileCatalogStore, tmp_path: Path
) -> None:
    store.replace_provider_models(
        provider=ProviderName.OPENAI,
        entries=[_entry(ProviderName.OPENAI, "gpt-4.1")],
    )
    assert (tmp_path / "openai.json").exists()


def test_read_filters_blacklisted_models(store: FileCatalogStore) -> None:
    store.replace_provider_models(
        provider=ProviderName.ANTHROPIC,
        entries=[
            _entry(ProviderName.ANTHROPIC, "claude-3-haiku-20240307"),
            _entry(ProviderName.ANTHROPIC, "claude-haiku-4-5-20251001"),
        ],
    )

    result = store.list_models(
        query=ModelCatalogQuery(provider=ProviderName.ANTHROPIC)
    )
    assert [entry.model for entry in result] == ["claude-haiku-4-5-20251001"]
    assert (
        store.count_models(
            query=ModelCatalogQuery(provider=ProviderName.ANTHROPIC)
        )
        == 1
    )
    assert (
        store.get_model(
            provider=ProviderName.ANTHROPIC, model="claude-3-haiku-20240307"
        )
        is None
    )


def test_load_all_skips_corrupt_cache_files_with_warning(
    store: FileCatalogStore, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store.replace_provider_models(
        provider=ProviderName.OPENAI,
        entries=[_entry(ProviderName.OPENAI, "gpt-4.1")],
    )
    bad = tmp_path / "anthropic.json"
    bad.write_text("not valid json {{{", encoding="utf-8")

    with caplog.at_level("WARNING", logger="dr_llm.llm.catalog.file_store"):
        all_models = store.list_models(query=ModelCatalogQuery())
    assert len(all_models) == 1
    assert all_models[0].provider == ProviderName.OPENAI
    assert any(
        "Skipping unreadable catalog cache file" in r.message
        for r in caplog.records
    )


def test_load_single_provider_still_raises_on_corrupt_file(
    tmp_path: Path,
) -> None:
    store = FileCatalogStore(cache_dir=tmp_path)
    (tmp_path / "anthropic.json").write_text(
        "not valid json", encoding="utf-8"
    )
    with pytest.raises(CatalogCacheCorruptError):
        store.list_models(
            query=ModelCatalogQuery(provider=ProviderName.ANTHROPIC)
        )
