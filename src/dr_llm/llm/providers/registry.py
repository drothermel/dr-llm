from __future__ import annotations

from threading import RLock

from dr_llm.llm.names import ProviderName
from dr_llm.llm.providers.anthropic.orchestrator import AnthropicOrchestrator
from dr_llm.llm.providers.anthropic.provider import AnthropicProvider
from dr_llm.llm.providers.api_config import APIProviderConfig
from dr_llm.llm.providers.config import ProviderAvailabilityStatus
from dr_llm.llm.providers.claude_code.orchestrator import (
    ClaudeCodeOrchestrator,
)
from dr_llm.llm.providers.claude_code.provider import (
    ClaudeCodeProvider,
)
from dr_llm.llm.providers.codex.orchestrator import (
    CodexOrchestrator,
)
from dr_llm.llm.providers.codex.provider import CodexProvider
from dr_llm.llm.providers.glm.orchestrator import (
    GlmOrchestrator,
)
from dr_llm.llm.providers.google.orchestrator import GoogleOrchestrator
from dr_llm.llm.providers.google.provider import GoogleProvider
from dr_llm.llm.providers.kimi_code.orchestrator import KimiCodeOrchestrator
from dr_llm.llm.providers.kimi_code.provider import KimiCodeProvider
from dr_llm.llm.providers.minimax.orchestrator import MiniMaxOrchestrator
from dr_llm.llm.providers.minimax.provider import MiniMaxProvider
from dr_llm.llm.providers.openai.orchestrator import (
    OpenAIOrchestrator,
)
from dr_llm.llm.providers.openai_compat_config import OpenAICompatConfig
from dr_llm.llm.providers.openai_compat_provider import OpenAICompatProvider
from dr_llm.llm.providers.openrouter.orchestrator import (
    OpenRouterOrchestrator,
)
from dr_llm.llm.providers.protocol import ProviderOrchestrator


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


_OPENAI_COMPAT_PROVIDERS: tuple[tuple[ProviderName, str, str], ...] = (
    (ProviderName.OPENAI, "https://api.openai.com/v1", "OPENAI_API_KEY"),
    (
        ProviderName.OPENROUTER,
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
    ),
    (ProviderName.GLM, "https://api.z.ai/api/coding/paas/v4", "ZAI_API_KEY"),
)

_OPENAI_COMPAT_ORCHESTRATORS = {
    ProviderName.OPENAI: OpenAIOrchestrator,
    ProviderName.OPENROUTER: OpenRouterOrchestrator,
    ProviderName.GLM: GlmOrchestrator,
}


def build_default_registry() -> ProviderRegistry:
    registry = ProviderRegistry()

    for name, base_url, api_key_env in _OPENAI_COMPAT_PROVIDERS:
        provider = OpenAICompatProvider(
            config=OpenAICompatConfig(
                name=name,
                base_url=base_url,
                api_key_env=api_key_env,
            ),
        )
        registry.register(_OPENAI_COMPAT_ORCHESTRATORS[name](provider))

    minimax = MiniMaxProvider()
    registry.register(MiniMaxOrchestrator(minimax))

    anthropic = AnthropicProvider()
    registry.register(AnthropicOrchestrator(anthropic))

    google = GoogleProvider(
        config=APIProviderConfig(
            name=ProviderName.GOOGLE,
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key_env="GOOGLE_API_KEY",
        )
    )
    registry.register(GoogleOrchestrator(google))

    codex = CodexProvider()
    registry.register(CodexOrchestrator(codex))

    claude_code = ClaudeCodeProvider()
    registry.register(ClaudeCodeOrchestrator(claude_code))

    kimi_code = KimiCodeProvider()
    registry.register(KimiCodeOrchestrator(kimi_code))

    return registry
