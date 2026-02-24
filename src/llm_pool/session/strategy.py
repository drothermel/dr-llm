from __future__ import annotations

from typing import Literal

from llm_pool.providers.base import ProviderCapabilities
from llm_pool.types import ToolPolicy


def resolve_tool_strategy(
    *, policy: ToolPolicy, capabilities: ProviderCapabilities
) -> Literal["brokered", "native"]:
    if policy == ToolPolicy.brokered_only:
        return "brokered"
    if policy == ToolPolicy.native_only:
        return "native"
    if capabilities.supports_native_tools:
        return "native"
    return "brokered"
