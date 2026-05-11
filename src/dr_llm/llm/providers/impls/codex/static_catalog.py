from __future__ import annotations

from enum import StrEnum


class CodexStaticCatalogModel(StrEnum):
    GPT54 = "gpt-5.4"
    GPT54_MINI = "gpt-5.4-mini"
    GPT53_CODEX = "gpt-5.3-codex"
    GPT53_CODEX_SPARK = "gpt-5.3-codex-spark"
    GPT52_CODEX = "gpt-5.2-codex"
    GPT52 = "gpt-5.2"
    GPT51_CODEX_MAX = "gpt-5.1-codex-max"
    GPT51_CODEX = "gpt-5.1-codex"
    GPT51 = "gpt-5.1"
    GPT5_CODEX = "gpt-5-codex"
    GPT51_CODEX_MINI = "gpt-5.1-codex-mini"
    GPT5 = "gpt-5"

    @classmethod
    def values(cls) -> list[str]:
        return [model.value for model in cls]
