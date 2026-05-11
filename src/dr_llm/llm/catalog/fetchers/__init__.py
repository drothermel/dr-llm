from __future__ import annotations

from dr_llm.llm.catalog.fetchers.anthropic import fetch_anthropic_models
from dr_llm.llm.catalog.fetchers.google import fetch_google_models
from dr_llm.llm.catalog.fetchers.kimi import fetch_kimi_models
from dr_llm.llm.catalog.fetchers.openai_compat import (
    fetch_openai_compat_models,
)
from dr_llm.llm.catalog.fetchers.static import (
    build_static_catalog_entries,
)


__all__ = [
    "fetch_anthropic_models",
    "fetch_google_models",
    "fetch_kimi_models",
    "fetch_openai_compat_models",
    "build_static_catalog_entries",
]
