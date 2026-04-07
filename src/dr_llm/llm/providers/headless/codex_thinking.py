from __future__ import annotations

CODEX_THINKING_SUPPORTED_MODELS = [
    "gpt-5",
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5.4",
    "gpt-5-codex",
    "gpt-5.1-codex",
    "gpt-5.1-codex-mini",
    "gpt-5.1-codex-max",
    "gpt-5.2-codex",
    "gpt-5.3-codex",
    "gpt-5.3-codex-spark",
    "gpt-5.4-mini",
]

CODEX_MINIMAL_THINKING_SUPPORTED_MODELS = [
    "gpt-5",
]

CODEX_OFF_THINKING_SUPPORTED_MODELS = [
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5.4",
    "gpt-5.4-mini",
]


def codex_supports_configurable_thinking(model: str) -> bool:
    return _matches_family(normalized=model, families=CODEX_THINKING_SUPPORTED_MODELS)


def codex_supports_minimal_thinking(model: str) -> bool:
    return _matches_family(
        normalized=model,
        families=CODEX_MINIMAL_THINKING_SUPPORTED_MODELS,
    )


def codex_supports_off_thinking(model: str) -> bool:
    return _matches_family(
        normalized=model,
        families=CODEX_OFF_THINKING_SUPPORTED_MODELS,
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
