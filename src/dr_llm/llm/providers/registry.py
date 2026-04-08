from __future__ import annotations

from threading import RLock

from dr_llm.llm.providers.anthropic.provider import AnthropicProvider
from dr_llm.llm.providers.api_config import APIProviderConfig
from dr_llm.llm.providers.base import Provider
from dr_llm.llm.providers.config import ProviderAvailabilityStatus
from dr_llm.llm.providers.google.provider import GoogleProvider
from dr_llm.llm.providers.headless.claude import ClaudeHeadlessProvider
from dr_llm.llm.providers.headless.codex import CodexHeadlessProvider
from dr_llm.llm.providers.kimi_code import KimiCodeProvider
from dr_llm.llm.providers.minimax import MiniMaxProvider
from dr_llm.llm.providers.openai_compat.config import OpenAICompatConfig
from dr_llm.llm.providers.openai_compat.provider import OpenAICompatProvider


class ProviderRegistry:
    def __init__(self) -> None:
        self._lock = RLock()
        self._providers: dict[str, Provider] = {}

    def register(self, provider: Provider) -> None:
        raw_primary = provider.name
        primary = raw_primary.strip()
        if not primary:
            raise ValueError("provider.name must be non-empty")
        if raw_primary != primary:
            raise ValueError(
                "provider.name must not have leading or trailing whitespace"
            )

        with self._lock:
            normalized_primary = primary.lower()
            existing = self._providers.get(normalized_primary)
            if existing is not None:
                raise ValueError(
                    f"register conflict for provider {primary!r}: {existing!r} is already registered"
                )
            self._providers[normalized_primary] = provider

    def get(self, provider_name: str) -> Provider:
        key = provider_name.strip().lower()
        with self._lock:
            provider = self._providers.get(key)
            if provider is None:
                known = ", ".join(sorted(self._providers.keys()))
                raise KeyError(
                    f"Unknown provider {provider_name!r}. Known providers: {known}"
                )
            return provider

    def names(self) -> set[str]:
        with self._lock:
            return set(self._providers.keys())

    def sorted_names(self) -> list[str]:
        return sorted(self.names())

    def availability_status(self, provider_name: str) -> ProviderAvailabilityStatus:
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
            providers = list(self._providers.values())
            self._providers.clear()
        for provider in providers:
            provider.close()


_OPENAI_COMPAT_PROVIDERS: tuple[tuple[str, str, str], ...] = (
    ("openai", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    ("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    ("glm", "https://api.z.ai/api/coding/paas/v4", "ZAI_API_KEY"),
)


def build_default_registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    for name, base_url, api_key_env in _OPENAI_COMPAT_PROVIDERS:
        registry.register(
            OpenAICompatProvider(
                config=OpenAICompatConfig(
                    name=name,
                    base_url=base_url,
                    api_key_env=api_key_env,
                ),
            )
        )
    registry.register(MiniMaxProvider())
    registry.register(AnthropicProvider())
    registry.register(
        GoogleProvider(
            config=APIProviderConfig(
                name="google",
                base_url="https://generativelanguage.googleapis.com/v1beta",
                api_key_env="GOOGLE_API_KEY",
            )
        )
    )
    registry.register(CodexHeadlessProvider())
    registry.register(ClaudeHeadlessProvider())
    registry.register(KimiCodeProvider())
    return registry
