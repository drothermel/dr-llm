from __future__ import annotations


class LlmPoolError(Exception):
    """Base error for llm_pool."""


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


class SessionConflictError(LlmPoolError):
    pass


class ToolExecutionError(LlmPoolError):
    pass


class ReplayError(LlmPoolError):
    pass
