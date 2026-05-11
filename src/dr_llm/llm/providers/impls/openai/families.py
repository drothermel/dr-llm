from __future__ import annotations

from enum import StrEnum


class OpenAIModelFamily(StrEnum):
    GPT5 = "gpt-5"
    GPT5_MINI = "gpt-5-mini"
    GPT5_NANO = "gpt-5-nano"
    GPT51 = "gpt-5.1"
    GPT51_MINI = "gpt-5.1-mini"
    GPT51_NANO = "gpt-5.1-nano"
    GPT51_CODEX = "gpt-5.1-codex"
    GPT51_CODEX_MINI = "gpt-5.1-codex-mini"
    GPT51_CODEX_MAX = "gpt-5.1-codex-max"
    GPT52 = "gpt-5.2"
    GPT52_MINI = "gpt-5.2-mini"
    GPT52_NANO = "gpt-5.2-nano"
    GPT52_CODEX = "gpt-5.2-codex"
    GPT53 = "gpt-5.3"
    GPT53_MINI = "gpt-5.3-mini"
    GPT53_NANO = "gpt-5.3-nano"
    GPT53_CODEX = "gpt-5.3-codex"
    GPT54 = "gpt-5.4"
    GPT54_MINI = "gpt-5.4-mini"
    GPT54_NANO = "gpt-5.4-nano"


OPENAI_GPT5_FAMILIES = (
    OpenAIModelFamily.GPT5,
    OpenAIModelFamily.GPT5_MINI,
    OpenAIModelFamily.GPT5_NANO,
)
OPENAI_GPT51_FAMILIES = (
    OpenAIModelFamily.GPT51,
    OpenAIModelFamily.GPT51_MINI,
    OpenAIModelFamily.GPT51_NANO,
    OpenAIModelFamily.GPT51_CODEX,
    OpenAIModelFamily.GPT51_CODEX_MINI,
    OpenAIModelFamily.GPT51_CODEX_MAX,
)
OPENAI_GPT52_FAMILIES = (
    OpenAIModelFamily.GPT52,
    OpenAIModelFamily.GPT52_MINI,
    OpenAIModelFamily.GPT52_NANO,
    OpenAIModelFamily.GPT52_CODEX,
)
OPENAI_GPT53_FAMILIES = (
    OpenAIModelFamily.GPT53,
    OpenAIModelFamily.GPT53_MINI,
    OpenAIModelFamily.GPT53_NANO,
    OpenAIModelFamily.GPT53_CODEX,
)
OPENAI_GPT54_FAMILIES = (
    OpenAIModelFamily.GPT54,
    OpenAIModelFamily.GPT54_MINI,
    OpenAIModelFamily.GPT54_NANO,
)
OPENAI_THINKING_SUPPORTED_MODELS = (
    *OPENAI_GPT5_FAMILIES,
    *OPENAI_GPT51_FAMILIES,
    *OPENAI_GPT52_FAMILIES,
    *OPENAI_GPT53_FAMILIES,
    *OPENAI_GPT54_FAMILIES,
)
OPENAI_MINIMAL_THINKING_SUPPORTED_MODELS = OPENAI_GPT5_FAMILIES
OPENAI_OFF_THINKING_SUPPORTED_MODELS = (
    *OPENAI_GPT51_FAMILIES,
    *OPENAI_GPT52_FAMILIES,
    *OPENAI_GPT53_FAMILIES,
    *OPENAI_GPT54_FAMILIES,
)
OPENAI_GPT5_SAMPLING_SUPPORTED_MODELS = (
    OpenAIModelFamily.GPT52,
    OpenAIModelFamily.GPT52_MINI,
    OpenAIModelFamily.GPT52_NANO,
    OpenAIModelFamily.GPT54,
    OpenAIModelFamily.GPT54_MINI,
    OpenAIModelFamily.GPT54_NANO,
)
