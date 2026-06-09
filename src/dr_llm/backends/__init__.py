"""Public backends API for programmatic LLM integration."""

from __future__ import annotations

from dr_llm.backends.direct import DirectBackend
from dr_llm.backends.errors import (
    BackendAcquireTimeoutError,
    BackendDrainTimeoutError,
    BackendError,
    BackendGenerationError,
    BackendSchemaError,
    BackendUnsupportedFeatureError,
    BackendValidationError,
)
from dr_llm.backends.fingerprint import fingerprint_request
from dr_llm.backends.models import (
    AcquireResult,
    BackendCapabilities,
    BackendRequest,
    BackendResponse,
    DrainResult,
    PoolBackendConfig,
    SubmitResult,
)
from dr_llm.backends.pool import PoolBackend

__all__ = [
    "AcquireResult",
    "BackendAcquireTimeoutError",
    "BackendCapabilities",
    "BackendDrainTimeoutError",
    "BackendError",
    "BackendGenerationError",
    "BackendRequest",
    "BackendResponse",
    "BackendSchemaError",
    "BackendUnsupportedFeatureError",
    "BackendValidationError",
    "DirectBackend",
    "DrainResult",
    "PoolBackend",
    "PoolBackendConfig",
    "SubmitResult",
    "fingerprint_request",
]
