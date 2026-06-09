"""Backend API error hierarchy."""

from __future__ import annotations


class BackendError(Exception):
    """Base error for dr_llm.backends operations."""


class BackendValidationError(BackendError):
    """Invalid backend request shape or content."""


class BackendUnsupportedFeatureError(BackendError):
    """Request uses a v1-unsupported extension (tools, multimodal, etc.)."""


class BackendAcquireTimeoutError(BackendError):
    """Session acquire did not claim enough samples before the timeout."""


class BackendDrainTimeoutError(BackendError):
    """Worker drain did not finish before the timeout."""


class BackendSchemaError(BackendError):
    """Pool schema is incompatible with PoolBackend expectations."""


class BackendGenerationError(BackendError):
    """Provider generation failed during pool top-up or fill."""
