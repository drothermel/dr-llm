from __future__ import annotations

from functools import cache
from importlib.resources import files

import yaml
from pydantic import BaseModel, ConfigDict

from dr_llm.llm.catalog.models import ModelCatalogEntry


class BlacklistedModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    reason: str


class OpenAIModelPrice(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_cost_per_1m: float
    output_cost_per_1m: float


@cache
def _blacklist() -> dict[tuple[str, str], str]:
    raw = yaml.safe_load(
        files("dr_llm.llm.catalog.data")
        .joinpath("model_blacklist.yml")
        .read_text(encoding="utf-8")
    )
    return {
        (provider, model): reason
        for provider, models in raw.items()
        for model, reason in models.items()
    }


@cache
def openai_language_model_pricing() -> dict[str, OpenAIModelPrice]:
    raw = yaml.safe_load(
        files("dr_llm.llm.catalog.data")
        .joinpath("openai_pricing.yml")
        .read_text(encoding="utf-8")
    )
    return {
        model: OpenAIModelPrice.model_validate(price) for model, price in raw.items()
    }


def blacklist_reason(*, provider: str, model: str) -> str | None:
    return _blacklist().get((provider, model))


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
    for (item_provider, item_model), reason in sorted(_blacklist().items()):
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
