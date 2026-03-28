from __future__ import annotations

from threading import RLock

from dr_llm.providers.base import ProviderAdapter, ProviderAvailabilityStatus


class ProviderRegistry:
    def __init__(self) -> None:
        self._lock = RLock()
        self._adapters: dict[str, ProviderAdapter] = {}

    def register(self, adapter: ProviderAdapter) -> None:
        raw_primary = adapter.name
        primary = raw_primary.strip()
        if not primary:
            raise ValueError("adapter.name must be non-empty")
        if raw_primary != primary:
            raise ValueError(
                "adapter.name must not have leading or trailing whitespace"
            )

        with self._lock:
            normalized_primary = primary.lower()
            existing = self._adapters.get(normalized_primary)
            if existing is not None:
                raise ValueError(
                    f"register conflict for provider {primary!r}: {existing!r} is already registered"
                )
            self._adapters[normalized_primary] = adapter

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

    def sorted_names(self) -> list[str]:
        return sorted(self.names())

    def availability_statuses(self) -> list[ProviderAvailabilityStatus]:
        return [
            self.get(provider_name).availability_status()
            for provider_name in self.sorted_names()
        ]

    def available_names(
        self,
        *,
        statuses: list[ProviderAvailabilityStatus] | None = None,
    ) -> list[str]:
        if statuses is None:
            statuses = self.availability_statuses()
        return [status.provider for status in statuses if status.available]

    def close(self) -> None:
        with self._lock:
            adapters = list(self._adapters.values())
            self._adapters.clear()
        for adapter in adapters:
            adapter.close()
