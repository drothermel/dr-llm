from __future__ import annotations


class LlmPoolError(Exception):
    """Base error for dr_llm."""


class ProviderError(LlmPoolError):
    pass


class ProviderTransportError(ProviderError):
    pass


class ProviderSemanticError(ProviderError):
    pass


class HeadlessExecutionError(ProviderError):
    pass


class PersistenceError(LlmPoolError):
    pass


class TransientPersistenceError(PersistenceError):
    pass
