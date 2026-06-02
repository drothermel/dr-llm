from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from dr_llm.llm import ProviderAvailabilityStatus, ProviderName


def _load_worker_demo() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "demo-streaming-log-worker.py"
    )
    spec = importlib.util.spec_from_file_location(
        "demo_streaming_log_worker", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _status(provider: str) -> ProviderAvailabilityStatus:
    return ProviderAvailabilityStatus(provider=provider, available=True)


def test_candidate_providers_prefer_configured_order() -> None:
    worker_demo = _load_worker_demo()

    providers = worker_demo._candidate_providers(
        [
            _status("anthropic"),
            _status("custom-provider"),
            _status("openai"),
        ],
        requested_provider=None,
    )

    assert providers == ["openai", "anthropic", "custom-provider"]


def test_candidate_providers_honor_requested_provider() -> None:
    worker_demo = _load_worker_demo()

    providers = worker_demo._candidate_providers(
        [_status("anthropic"), _status("openai")],
        requested_provider=ProviderName.ANTHROPIC,
    )

    assert providers == ["anthropic"]


def test_candidate_providers_reject_unavailable_requested_provider() -> None:
    worker_demo = _load_worker_demo()

    with pytest.raises(RuntimeError, match="Requested provider 'anthropic'"):
        worker_demo._candidate_providers(
            [_status("openai")],
            requested_provider=ProviderName.ANTHROPIC,
        )
