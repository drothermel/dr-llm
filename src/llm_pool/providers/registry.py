from __future__ import annotations

from threading import RLock

from llm_pool.providers.base import ProviderAdapter


class ProviderRegistry:
    def __init__(self) -> None:
        self._lock = RLock()
        self._adapters: dict[str, ProviderAdapter] = {}

    def register(
        self, adapter: ProviderAdapter, *, aliases: list[str] | None = None
    ) -> None:
        raw_primary = adapter.name
        primary = raw_primary.strip()
        if not primary:
            raise ValueError("adapter.name must be non-empty")
        if raw_primary != primary:
            raise ValueError(
                "adapter.name must not have leading or trailing whitespace"
            )

        names = [primary]
        if aliases:
            for alias in aliases:
                normalized = alias.strip()
                if not normalized:
                    raise ValueError("provider alias must be non-empty")
                if alias != normalized:
                    raise ValueError(
                        "provider alias must not have leading or trailing whitespace"
                    )
                names.append(normalized)
        with self._lock:
            for name in names:
                self._adapters[name.lower()] = adapter

    def get(self, provider_name: str) -> ProviderAdapter:
        key = provider_name.strip().lower()
        with self._lock:
            adapter = self._adapters.get(key)
            known = ", ".join(sorted(self._adapters.keys()))
        if adapter is None:
            raise KeyError(
                f"Unknown provider {provider_name!r}. Known providers: {known}"
            )
        return adapter

    def names(self) -> set[str]:
        with self._lock:
            return set(self._adapters.keys())
