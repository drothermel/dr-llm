from __future__ import annotations

from enum import StrEnum


class CodexModelFamily(StrEnum):
    GPT5 = "gpt-5"
    GPT51 = "gpt-5.1"
    GPT52 = "gpt-5.2"
    GPT54 = "gpt-5.4"
    GPT5_CODEX = "gpt-5-codex"
    GPT51_CODEX = "gpt-5.1-codex"
    GPT51_CODEX_MINI = "gpt-5.1-codex-mini"
    GPT51_CODEX_MAX = "gpt-5.1-codex-max"
    GPT52_CODEX = "gpt-5.2-codex"
    GPT53_CODEX = "gpt-5.3-codex"
    GPT53_CODEX_SPARK = "gpt-5.3-codex-spark"
    GPT54_MINI = "gpt-5.4-mini"


CODEX_THINKING_SUPPORTED_MODELS = (
    CodexModelFamily.GPT5,
    CodexModelFamily.GPT51,
    CodexModelFamily.GPT52,
    CodexModelFamily.GPT54,
    CodexModelFamily.GPT5_CODEX,
    CodexModelFamily.GPT51_CODEX,
    CodexModelFamily.GPT51_CODEX_MINI,
    CodexModelFamily.GPT51_CODEX_MAX,
    CodexModelFamily.GPT52_CODEX,
    CodexModelFamily.GPT53_CODEX,
    CodexModelFamily.GPT53_CODEX_SPARK,
    CodexModelFamily.GPT54_MINI,
)
CODEX_MINIMAL_THINKING_SUPPORTED_MODELS = (CodexModelFamily.GPT5,)
CODEX_OFF_THINKING_SUPPORTED_MODELS = (
    CodexModelFamily.GPT51,
    CodexModelFamily.GPT52,
    CodexModelFamily.GPT54,
    CodexModelFamily.GPT54_MINI,
)
