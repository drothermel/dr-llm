from __future__ import annotations

OPENAI_THINKING_SUPPORTED_MODELS = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5.1",
    "gpt-5.1-mini",
    "gpt-5.1-nano",
    "gpt-5.1-codex",
    "gpt-5.1-codex-mini",
    "gpt-5.1-codex-max",
    "gpt-5.2",
    "gpt-5.2-mini",
    "gpt-5.2-nano",
    "gpt-5.2-codex",
    "gpt-5.3",
    "gpt-5.3-mini",
    "gpt-5.3-nano",
    "gpt-5.3-codex",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
]

OPENAI_MINIMAL_THINKING_SUPPORTED_MODELS = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
]

OPENAI_OFF_THINKING_SUPPORTED_MODELS = [
    "gpt-5.1",
    "gpt-5.1-mini",
    "gpt-5.1-nano",
    "gpt-5.1-codex",
    "gpt-5.1-codex-mini",
    "gpt-5.1-codex-max",
    "gpt-5.2",
    "gpt-5.2-mini",
    "gpt-5.2-nano",
    "gpt-5.2-codex",
    "gpt-5.3",
    "gpt-5.3-mini",
    "gpt-5.3-nano",
    "gpt-5.3-codex",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
]

OPENAI_XHIGH_THINKING_SUPPORTED_MODELS = [
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
]


def normalize_openai_reasoning_model(model: str) -> str:
    if model.startswith("openai/"):
        return model[len("openai/") :]
    return model


def openai_supports_configurable_thinking(model: str) -> bool:
    normalized = normalize_openai_reasoning_model(model)
    return _matches_family(
        normalized=normalized,
        families=OPENAI_THINKING_SUPPORTED_MODELS,
    )


def openai_supports_minimal_thinking(model: str) -> bool:
    normalized = normalize_openai_reasoning_model(model)
    return _matches_family(
        normalized=normalized,
        families=OPENAI_MINIMAL_THINKING_SUPPORTED_MODELS,
    )


def openai_supports_off_thinking(model: str) -> bool:
    normalized = normalize_openai_reasoning_model(model)
    return _matches_family(
        normalized=normalized,
        families=OPENAI_OFF_THINKING_SUPPORTED_MODELS,
    )


def openai_supports_xhigh_thinking(model: str) -> bool:
    normalized = normalize_openai_reasoning_model(model)
    return _matches_family(
        normalized=normalized,
        families=OPENAI_XHIGH_THINKING_SUPPORTED_MODELS,
    )


def _matches_family(*, normalized: str, families: list[str]) -> bool:
    return any(
        normalized == family
        or _is_snapshot_of_family(normalized=normalized, family=family)
        for family in families
    )


def _is_snapshot_of_family(*, normalized: str, family: str) -> bool:
    prefix = f"{family}-"
    if not normalized.startswith(prefix):
        return False
    suffix = normalized[len(prefix) :]
    return bool(suffix) and suffix[0].isdigit()
