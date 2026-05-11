from __future__ import annotations

from enum import StrEnum


class OpenAIStaticCatalogModel(StrEnum):
    GPT54 = "gpt-5.4"
    GPT54_MINI = "gpt-5.4-mini"
    GPT53 = "gpt-5.3"
    GPT52 = "gpt-5.2"
    GPT51 = "gpt-5.1"
    GPT5 = "gpt-5"
    O3 = "o3"
    O3_MINI = "o3-mini"
    O4_MINI = "o4-mini"

    @classmethod
    def values(cls) -> list[str]:
        return [model.value for model in cls]
