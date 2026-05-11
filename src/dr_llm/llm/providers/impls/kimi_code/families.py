from __future__ import annotations

from enum import StrEnum


class KimiCodeModelFamily(StrEnum):
    KIMI_FOR_CODING = "kimi-for-coding"


KIMI_CODE_SUPPORTED_MODELS = (KimiCodeModelFamily.KIMI_FOR_CODING,)
