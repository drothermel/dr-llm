from __future__ import annotations

from threading import RLock

from dr_llm.llm.providers.core.config import ProviderAvailabilityStatus
from dr_llm.llm.providers.core.protocol import ProviderOrchestrator


class ProviderRegistry:
    def __init__(self) -> None:
        self._lock = RLock()
        self._orchestrators: dict[str, ProviderOrchestrator] = {}

    def register(self, orchestrator: ProviderOrchestrator) -> None:
        raw_primary = orchestrator.name
        primary = raw_primary.strip()
        if not primary:
            raise ValueError("orchestrator.name must be non-empty")
        if raw_primary != primary:
            raise ValueError(
                "orchestrator.name must not have leading or trailing whitespace"
            )

        with self._lock:
            normalized_primary = primary.lower()
            existing = self._orchestrators.get(normalized_primary)
            if existing is not None:
                raise ValueError(
                    f"register conflict for provider {primary!r}: {existing!r} is already registered"
                )
            self._orchestrators[normalized_primary] = orchestrator

    def get(self, provider_name: str) -> ProviderOrchestrator:
        key = provider_name.strip().lower()
        with self._lock:
            orchestrator = self._orchestrators.get(key)
            if orchestrator is None:
                known = ", ".join(sorted(self._orchestrators.keys()))
                raise KeyError(
                    f"Unknown provider {provider_name!r}. Known providers: {known}"
                )
            return orchestrator

    def names(self) -> set[str]:
        with self._lock:
            return set(self._orchestrators.keys())

    def sorted_names(self) -> list[str]:
        return sorted(self.names())

    def availability_status(
        self, provider_name: str
    ) -> ProviderAvailabilityStatus:
        return self.get(provider_name).availability_status()

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
            orchestrators = list(self._orchestrators.values())
            self._orchestrators.clear()
        for orchestrator in orchestrators:
            orchestrator.close()
