from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from dr_llm.catalog.models import ModelCatalogEntry


class BlacklistedModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    reason: str


class OpenAIModelPrice(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_cost_per_1m: float
    output_cost_per_1m: float


IRRELEVANT_TO_LLM_RESEARCH = "Irrelevant to LLM research."
IRRELEVANT_FOR_RESEARCH = "Irrelevant for Research"
AVOID_MORE_EXPENSIVE_BUT_FASTER_MODELS = (
    "Avoid calling more expensive but faster models."
)

OPENAI_IRRELEVANT_MODELS: tuple[str, ...] = (
    "davinci-002",
    "babbage-002",
    "dall-e-3",
    "dall-e-2",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-instruct",
    "gpt-3.5-turbo-instruct-0914",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "tts-1-hd",
    "tts-1-1106",
    "tts-1-hd-1106",
    "text-embedding-3-small",
    "text-embedding-3-large",
    "gpt-4o-audio-preview",
    "gpt-4o-realtime-preview",
    "omni-moderation-latest",
    "omni-moderation-2024-09-26",
    "gpt-4o-realtime-preview-2024-12-17",
    "gpt-4o-audio-preview-2024-12-17",
    "gpt-4o-mini-realtime-preview-2024-12-17",
    "gpt-4o-mini-audio-preview-2024-12-17",
    "gpt-4o-mini-realtime-preview",
    "gpt-4o-mini-audio-preview",
    "gpt-4o-realtime-preview-2025-06-03",
    "gpt-4o-audio-preview-2025-06-03",
    "gpt-4o-transcribe-diarize",
    "gpt-4o-mini-search-preview-2025-03-11",
    "gpt-4o-mini-search-preview",
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
    "o1-pro-2025-03-19",
    "o1-pro",
    "gpt-4o-mini-tts",
    "gpt-image-1",
    "gpt-audio-2025-08-28",
    "gpt-realtime",
    "gpt-realtime-2025-08-28",
    "gpt-audio",
    "gpt-image-1-mini",
    "gpt-5-pro-2025-10-06",
    "gpt-5-pro",
    "gpt-audio-mini",
    "gpt-audio-mini-2025-10-06",
    "gpt-5-search-api",
    "gpt-realtime-mini",
    "gpt-realtime-mini-2025-10-06",
    "sora-2",
    "sora-2-pro",
    "gpt-5-search-api-2025-10-14",
    "gpt-5.1-chat-latest",
    "gpt-image-1.5",
    "gpt-5.2-chat-latest",
    "gpt-4o-mini-transcribe-2025-12-15",
    "gpt-4o-mini-transcribe-2025-03-20",
    "gpt-4o-mini-tts-2025-03-20",
    "gpt-4o-mini-tts-2025-12-15",
    "gpt-realtime-mini-2025-12-15",
    "gpt-audio-mini-2025-12-15",
    "chatgpt-image-latest",
    "gpt-5.2-pro-2025-12-11",
    "gpt-5.2-pro",
    "gpt-5.4-pro",
    "gpt-5.4-pro-2026-03-05",
    "gpt-3.5-turbo-16k",
    "tts-1",
    "whisper-1",
    "text-embedding-ada-002",
    "gpt-realtime-1.5",
    "gpt-audio-1.5",
    "gpt-4o-search-preview",
    "gpt-4o-search-preview-2025-03-11",
    "gpt-5.3-chat-latest",
)

GOOGLE_IRRELEVANT_MODELS: tuple[str, ...] = (
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-pro-latest",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash-image",
    "gemini-3.1-pro-preview-customtools",
    "gemini-3-pro-image-preview",
    "nano-banana-pro-preview",
    "gemini-3.1-flash-image-preview",
    "lyria-3-clip-preview",
    "lyria-3-pro-preview",
    "gemini-robotics-er-1.5-preview",
    "gemini-2.5-computer-use-preview-10-2025",
    "deep-research-pro-preview-12-2025",
    "gemini-embedding-001",
    "gemini-embedding-2-preview",
    "aqa",
    "imagen-4.0-generate-001",
    "imagen-4.0-ultra-generate-001",
    "imagen-4.0-fast-generate-001",
    "veo-2.0-generate-001",
    "veo-3.0-generate-001",
    "veo-3.0-fast-generate-001",
    "veo-3.1-generate-preview",
    "veo-3.1-fast-generate-preview",
    "veo-3.1-lite-generate-preview",
    "gemini-2.5-flash-native-audio-latest",
    "gemini-2.5-flash-native-audio-preview-09-2025",
    "gemini-2.5-flash-native-audio-preview-12-2025",
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro-preview-tts",
    "gemini-3.1-flash-live-preview",
    "gemini-3.1-pro-preview",
    "gemini-3-pro-preview",
    "gemini-2.5-pro",
)


MODEL_BLACKLIST: dict[tuple[str, str], str] = {
    (
        "anthropic",
        "claude-3-haiku-20240307",
    ): (
        "Deprecated by Anthropic on 2026-02-19 and scheduled to retire on 2026-04-20. "
        "Recommended replacement: claude-haiku-4-5-20251001."
    ),
    ("glm", "glm-5-turbo"): AVOID_MORE_EXPENSIVE_BUT_FASTER_MODELS,
    **{
        ("openai", model): IRRELEVANT_TO_LLM_RESEARCH
        for model in OPENAI_IRRELEVANT_MODELS
    },
    **{
        ("google", model): IRRELEVANT_FOR_RESEARCH for model in GOOGLE_IRRELEVANT_MODELS
    },
}


# Verified against official OpenAI model docs and pricing on 2026-04-03.
# OpenAI has not published public API token pricing for `gpt-5.3-codex-spark`
# as of 2026-04-03, so it is intentionally omitted from this table.
OPENAI_LANGUAGE_MODEL_PRICING: dict[str, OpenAIModelPrice] = {
    "gpt-5.4-2026-03-05": OpenAIModelPrice(
        input_cost_per_1m=2.5,
        output_cost_per_1m=15.0,
    ),
    "gpt-5.4": OpenAIModelPrice(
        input_cost_per_1m=2.5,
        output_cost_per_1m=15.0,
    ),
    "gpt-5.2-2025-12-11": OpenAIModelPrice(
        input_cost_per_1m=1.75,
        output_cost_per_1m=14.0,
    ),
    "gpt-5.2": OpenAIModelPrice(
        input_cost_per_1m=1.75,
        output_cost_per_1m=14.0,
    ),
    "gpt-5.1-2025-11-13": OpenAIModelPrice(
        input_cost_per_1m=1.25,
        output_cost_per_1m=10.0,
    ),
    "gpt-5.1": OpenAIModelPrice(
        input_cost_per_1m=1.25,
        output_cost_per_1m=10.0,
    ),
    "gpt-5-chat-latest": OpenAIModelPrice(
        input_cost_per_1m=1.25,
        output_cost_per_1m=10.0,
    ),
    "gpt-5-2025-08-07": OpenAIModelPrice(
        input_cost_per_1m=1.25,
        output_cost_per_1m=10.0,
    ),
    "gpt-5": OpenAIModelPrice(
        input_cost_per_1m=1.25,
        output_cost_per_1m=10.0,
    ),
    "gpt-4.1-2025-04-14": OpenAIModelPrice(
        input_cost_per_1m=2.0,
        output_cost_per_1m=8.0,
    ),
    "gpt-4.1": OpenAIModelPrice(
        input_cost_per_1m=2.0,
        output_cost_per_1m=8.0,
    ),
    "gpt-4o-2024-11-20": OpenAIModelPrice(
        input_cost_per_1m=2.5,
        output_cost_per_1m=10.0,
    ),
    "gpt-4o-2024-05-13": OpenAIModelPrice(
        input_cost_per_1m=2.5,
        output_cost_per_1m=10.0,
    ),
    "gpt-4o-2024-08-06": OpenAIModelPrice(
        input_cost_per_1m=2.5,
        output_cost_per_1m=10.0,
    ),
    "gpt-4o": OpenAIModelPrice(
        input_cost_per_1m=2.5,
        output_cost_per_1m=10.0,
    ),
    "gpt-4-0613": OpenAIModelPrice(
        input_cost_per_1m=30.0,
        output_cost_per_1m=60.0,
    ),
    "gpt-4": OpenAIModelPrice(
        input_cost_per_1m=30.0,
        output_cost_per_1m=60.0,
    ),
    "o3-2025-04-16": OpenAIModelPrice(
        input_cost_per_1m=2.0,
        output_cost_per_1m=8.0,
    ),
    "o3": OpenAIModelPrice(
        input_cost_per_1m=2.0,
        output_cost_per_1m=8.0,
    ),
    "o1-2024-12-17": OpenAIModelPrice(
        input_cost_per_1m=15.0,
        output_cost_per_1m=60.0,
    ),
    "o1": OpenAIModelPrice(
        input_cost_per_1m=15.0,
        output_cost_per_1m=60.0,
    ),
    "gpt-5.3-codex": OpenAIModelPrice(
        input_cost_per_1m=1.75,
        output_cost_per_1m=14.0,
    ),
    "gpt-5.2-codex": OpenAIModelPrice(
        input_cost_per_1m=1.75,
        output_cost_per_1m=14.0,
    ),
    "gpt-5.1-codex-max": OpenAIModelPrice(
        input_cost_per_1m=1.25,
        output_cost_per_1m=10.0,
    ),
    "gpt-5.1-codex": OpenAIModelPrice(
        input_cost_per_1m=1.25,
        output_cost_per_1m=10.0,
    ),
    "gpt-5-codex": OpenAIModelPrice(
        input_cost_per_1m=1.25,
        output_cost_per_1m=10.0,
    ),
    "gpt-5.1-codex-mini": OpenAIModelPrice(
        input_cost_per_1m=0.25,
        output_cost_per_1m=2.0,
    ),
    "gpt-5.4-mini-2026-03-17": OpenAIModelPrice(
        input_cost_per_1m=0.75,
        output_cost_per_1m=4.5,
    ),
    "gpt-5.4-mini": OpenAIModelPrice(
        input_cost_per_1m=0.75,
        output_cost_per_1m=4.5,
    ),
    "gpt-5-mini-2025-08-07": OpenAIModelPrice(
        input_cost_per_1m=0.25,
        output_cost_per_1m=2.0,
    ),
    "gpt-5-mini": OpenAIModelPrice(
        input_cost_per_1m=0.25,
        output_cost_per_1m=2.0,
    ),
    "gpt-4.1-mini-2025-04-14": OpenAIModelPrice(
        input_cost_per_1m=0.4,
        output_cost_per_1m=1.6,
    ),
    "gpt-4.1-mini": OpenAIModelPrice(
        input_cost_per_1m=0.4,
        output_cost_per_1m=1.6,
    ),
    "gpt-4o-mini-2024-07-18": OpenAIModelPrice(
        input_cost_per_1m=0.15,
        output_cost_per_1m=0.6,
    ),
    "gpt-4o-mini": OpenAIModelPrice(
        input_cost_per_1m=0.15,
        output_cost_per_1m=0.6,
    ),
    "o4-mini-2025-04-16": OpenAIModelPrice(
        input_cost_per_1m=1.1,
        output_cost_per_1m=4.4,
    ),
    "o4-mini": OpenAIModelPrice(
        input_cost_per_1m=1.1,
        output_cost_per_1m=4.4,
    ),
    "o3-mini": OpenAIModelPrice(
        input_cost_per_1m=1.1,
        output_cost_per_1m=4.4,
    ),
    "o3-mini-2025-01-31": OpenAIModelPrice(
        input_cost_per_1m=1.1,
        output_cost_per_1m=4.4,
    ),
    "gpt-5.4-nano-2026-03-17": OpenAIModelPrice(
        input_cost_per_1m=0.2,
        output_cost_per_1m=1.25,
    ),
    "gpt-5.4-nano": OpenAIModelPrice(
        input_cost_per_1m=0.2,
        output_cost_per_1m=1.25,
    ),
    "gpt-5-nano-2025-08-07": OpenAIModelPrice(
        input_cost_per_1m=0.05,
        output_cost_per_1m=0.4,
    ),
    "gpt-5-nano": OpenAIModelPrice(
        input_cost_per_1m=0.05,
        output_cost_per_1m=0.4,
    ),
    "gpt-4.1-nano-2025-04-14": OpenAIModelPrice(
        input_cost_per_1m=0.1,
        output_cost_per_1m=0.4,
    ),
    "gpt-4.1-nano": OpenAIModelPrice(
        input_cost_per_1m=0.1,
        output_cost_per_1m=0.4,
    ),
}


def blacklist_reason(*, provider: str, model: str) -> str | None:
    return MODEL_BLACKLIST.get((provider, model))


def filter_blacklisted_entries(
    entries: list[ModelCatalogEntry],
) -> list[ModelCatalogEntry]:
    return [
        entry
        for entry in entries
        if blacklist_reason(provider=entry.provider, model=entry.model) is None
    ]


def blacklisted_models(
    *, provider: str | None = None
) -> dict[str, list[BlacklistedModel]]:
    grouped: dict[str, list[BlacklistedModel]] = {}
    for (item_provider, item_model), reason in sorted(MODEL_BLACKLIST.items()):
        if provider is not None and item_provider != provider:
            continue
        grouped.setdefault(item_provider, []).append(
            BlacklistedModel(
                provider=item_provider,
                model=item_model,
                reason=reason,
            )
        )
    return grouped
