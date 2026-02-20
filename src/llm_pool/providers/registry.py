from __future__ import annotations

from threading import RLock

from llm_pool.providers.base import ProviderAdapter


class ProviderRegistry:
    def __init__(self) -> None:
        self._lock = RLock()
        self._adapters: dict[str, ProviderAdapter] = {}

    def register(self, adapter: ProviderAdapter, *, aliases: list[str] | None = None) -> None:
        names = [adapter.name]
        if aliases:
            names.extend(aliases)
        with self._lock:
            for name in names:
                self._adapters[name.strip().lower()] = adapter

    def get(self, provider_name: str) -> ProviderAdapter:
        key = provider_name.strip().lower()
        with self._lock:
            adapter = self._adapters.get(key)
        if adapter is None:
            known = ", ".join(sorted(self.names()))
            raise KeyError(f"Unknown provider {provider_name!r}. Known providers: {known}")
        return adapter

    def names(self) -> set[str]:
        with self._lock:
            return set(self._adapters.keys())
