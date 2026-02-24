from llm_pool.providers.base import ProviderCapabilities
from llm_pool.session.strategy import resolve_tool_strategy
from llm_pool.types import ToolPolicy


def test_strategy_native_preferred_with_native_capability() -> None:
    strategy = resolve_tool_strategy(
        policy=ToolPolicy.native_preferred,
        capabilities=ProviderCapabilities(supports_native_tools=True),
    )
    assert strategy == "native"


def test_strategy_native_preferred_without_native_capability() -> None:
    strategy = resolve_tool_strategy(
        policy=ToolPolicy.native_preferred,
        capabilities=ProviderCapabilities(supports_native_tools=False),
    )
    assert strategy == "brokered"


def test_strategy_brokered_only() -> None:
    strategy = resolve_tool_strategy(
        policy=ToolPolicy.brokered_only,
        capabilities=ProviderCapabilities(supports_native_tools=True),
    )
    assert strategy == "brokered"


def test_strategy_native_only() -> None:
    strategy = resolve_tool_strategy(
        policy=ToolPolicy.native_only,
        capabilities=ProviderCapabilities(supports_native_tools=False),
    )
    assert strategy == "native"
