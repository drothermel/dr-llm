from __future__ import annotations

from typing import Any, cast

from dr_llm.pool.pool_fill import seed_pending
from dr_llm.pool.sample_store import PoolStore
from dr_llm.providers.llm_config import LlmConfig
from dr_llm.providers.models import Message
from dr_llm.providers.reasoning import AnthropicReasoning, ThinkingLevel


class _FakeSchema:
    def __init__(self, key_column_names: list[str]) -> None:
        self.key_column_names = key_column_names


class _FakePendingStore:
    def __init__(self) -> None:
        self.samples: list[dict[str, Any]] = []
        self._seen: set[tuple[tuple[str, Any], ...]] = set()

    def insert_pending(self, sample: Any, *, ignore_conflicts: bool = True) -> bool:
        key = tuple(sorted(sample.key_values.items())) + (
            ("sample_idx", sample.sample_idx),
        )
        if ignore_conflicts and key in self._seen:
            return False
        self._seen.add(key)
        self.samples.append(
            {
                "key_values": dict(sample.key_values),
                "sample_idx": sample.sample_idx,
                "payload": dict(sample.payload),
                "priority": sample.priority,
            }
        )
        return True


class _FakeStore:
    def __init__(self, key_column_names: list[str]) -> None:
        self.schema = _FakeSchema(key_column_names)
        self.pending = _FakePendingStore()

    def init_schema(self) -> None:
        pass


def test_dict_grid_stores_ids_in_key_values_and_values_in_payload() -> None:
    store = _FakeStore(["llm_config", "prompt"])
    typed_store = cast(PoolStore, store)

    config_a = LlmConfig(provider="openai", model="gpt-4.1-mini")
    config_b = LlmConfig(
        provider="anthropic",
        model="claude-sonnet-4-6-20250514",
        max_tokens=256,
        reasoning=AnthropicReasoning(thinking_level=ThinkingLevel.OFF),
    )
    messages_x = [Message(role="user", content="Hello")]
    messages_y = [Message(role="user", content="Bye")]

    result = seed_pending(
        typed_store,
        key_grid={
            "llm_config": {"cfg_a": config_a, "cfg_b": config_b},
            "prompt": {"px": messages_x, "py": messages_y},
        },
        n=1,
    )

    assert result.inserted == 4
    sample = store.pending.samples[0]
    assert sample["key_values"] == {"llm_config": "cfg_a", "prompt": "px"}
    assert sample["payload"]["llm_config"] == config_a.model_dump()
    assert sample["payload"]["prompt"] == [m.model_dump() for m in messages_x]


def test_mixed_dict_and_plain_grid() -> None:
    store = _FakeStore(["llm_config", "tag"])
    typed_store = cast(PoolStore, store)

    config = LlmConfig(provider="openai", model="gpt-4.1-mini")
    result = seed_pending(
        typed_store,
        key_grid={
            "llm_config": {"cfg": config},
            "tag": ["v1", "v2"],
        },
        n=1,
    )

    assert result.inserted == 2
    s0 = store.pending.samples[0]
    assert s0["key_values"] == {"llm_config": "cfg", "tag": "v1"}
    assert s0["payload"] == {"llm_config": config.model_dump()}
    assert "tag" not in s0["payload"]

    s1 = store.pending.samples[1]
    assert s1["key_values"]["tag"] == "v2"


def test_plain_dict_values_serialized_as_is() -> None:
    store = _FakeStore(["dim"])
    typed_store = cast(PoolStore, store)

    result = seed_pending(
        typed_store,
        key_grid={"dim": {"id1": {"raw": "data"}}},
        n=1,
    )

    assert result.inserted == 1
    assert store.pending.samples[0]["payload"]["dim"] == {"raw": "data"}


def test_empty_dict_grid_returns_zero_inserted() -> None:
    store = _FakeStore(["dim"])
    typed_store = cast(PoolStore, store)

    result = seed_pending(
        typed_store,
        key_grid={"dim": {}},
        n=3,
    )

    assert result.inserted == 0
    assert len(store.pending.samples) == 0


def test_dict_grid_with_n_greater_than_one() -> None:
    store = _FakeStore(["cfg"])
    typed_store = cast(PoolStore, store)

    config = LlmConfig(provider="openai", model="gpt-4.1-mini")
    result = seed_pending(
        typed_store,
        key_grid={"cfg": {"c1": config}},
        n=3,
    )

    assert result.inserted == 3
    indices = [s["sample_idx"] for s in store.pending.samples]
    assert indices == [0, 1, 2]
    assert all(s["payload"]["cfg"] == config.model_dump() for s in store.pending.samples)
