import asyncio

import pytest

from llm_pool.tools.executor import ToolExecutor
from llm_pool.tools.registry import ToolDefinition, ToolRegistry
from llm_pool.types import ProviderToolSpec, ToolErrorCode, ToolInvocation


def test_tool_executor_sync_handler() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="add",
            description="Add two ints",
            input_schema={"type": "object"},
            handler=lambda args: {"sum": int(args["a"]) + int(args["b"])},
        )
    )
    executor = ToolExecutor(registry=registry)
    result = executor.invoke(
        ToolInvocation(
            tool_call_id="tc1", name="add", arguments={"a": 2, "b": 3}, session_id="s1"
        )
    )
    assert result.ok
    assert result.result == {"sum": 5}


def test_tool_executor_unknown_tool() -> None:
    executor = ToolExecutor(registry=ToolRegistry())
    result = executor.invoke(
        ToolInvocation(
            tool_call_id="tc1", name="missing", arguments={}, session_id="s1"
        )
    )
    assert not result.ok
    assert result.error is not None
    assert result.error.error_code == ToolErrorCode.unknown_tool


def test_tool_executor_async_handler_with_running_loop_returns_error() -> None:
    async def async_handler(args: dict[str, int]) -> dict[str, int]:
        return {"value": args["x"]}

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="async_tool",
            description="Returns value",
            input_schema={"type": "object"},
            handler=async_handler,
        )
    )
    executor = ToolExecutor(registry=registry)

    async def invoke_inside_loop():
        return executor.invoke(
            ToolInvocation(
                tool_call_id="tc1",
                name="async_tool",
                arguments={"x": 7},
                session_id="s1",
            )
        )

    result = asyncio.run(invoke_inside_loop())
    assert not result.ok
    assert result.error is not None
    assert result.error.error_code == ToolErrorCode.tool_async_in_running_loop


def test_registry_to_provider_tools_returns_typed_models() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="lookup",
            description="Lookup by key",
            input_schema={
                "type": "object",
                "properties": {"k": {"type": "string"}},
                "required": ["k"],
            },
            handler=lambda args: {"k": args.get("k")},
        )
    )
    tools = registry.to_provider_tools()
    assert len(tools) == 1
    assert isinstance(tools[0], ProviderToolSpec)
    assert tools[0].function.name == "lookup"


def test_tool_executor_invoke_async_with_async_handler() -> None:
    async def async_handler(args: dict[str, int]) -> dict[str, int]:
        return {"value": args["x"]}

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="async_tool",
            description="Returns value",
            input_schema={"type": "object"},
            handler=async_handler,
        )
    )
    executor = ToolExecutor(registry=registry)

    async def run_invoke_async():
        return await executor.invoke_async(
            ToolInvocation(
                tool_call_id="tc_async",
                name="async_tool",
                arguments={"x": 9},
                session_id="s1",
            )
        )

    result = asyncio.run(run_invoke_async())
    assert result.ok
    assert result.result == {"value": 9}


def test_tool_executor_invoke_async_wraps_non_dict_results() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="plain",
            description="Returns scalar",
            input_schema={"type": "object"},
            handler=lambda _args: 42,
        )
    )
    executor = ToolExecutor(registry=registry)

    async def run_invoke_async():
        return await executor.invoke_async(
            ToolInvocation(
                tool_call_id="tc_plain",
                name="plain",
                arguments={},
                session_id="s1",
            )
        )

    result = asyncio.run(run_invoke_async())
    assert result.ok
    assert result.result == {"value": 42}


def test_tool_executor_invoke_async_unknown_tool() -> None:
    executor = ToolExecutor(registry=ToolRegistry())

    async def run_invoke_async():
        return await executor.invoke_async(
            ToolInvocation(
                tool_call_id="tc_missing",
                name="missing",
                arguments={},
                session_id="s1",
            )
        )

    result = asyncio.run(run_invoke_async())
    assert not result.ok
    assert result.error is not None
    assert result.error.error_code == ToolErrorCode.unknown_tool


def test_tool_executor_invoke_async_handler_exception() -> None:
    def failing_handler(_args):
        raise ValueError("boom")

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="failing",
            description="Raises",
            input_schema={"type": "object"},
            handler=failing_handler,
        )
    )
    executor = ToolExecutor(registry=registry)

    async def run_invoke_async():
        return await executor.invoke_async(
            ToolInvocation(
                tool_call_id="tc_fail",
                name="failing",
                arguments={},
                session_id="s1",
            )
        )

    result = asyncio.run(run_invoke_async())
    assert not result.ok
    assert result.error is not None
    assert result.error.error_code == ToolErrorCode.tool_execution_failed
    assert result.error.exception_type == "ValueError"
    assert result.error.message == "boom"


def test_tool_registry_get_is_case_insensitive() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="Lookup",
            description="Case test",
            input_schema={"type": "object"},
            handler=lambda args: {"q": args.get("q")},
        )
    )

    tool = registry.get("lookup")
    assert tool.name == "Lookup"


def test_tool_registry_rejects_case_collisions() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="Lookup",
            description="First",
            input_schema={"type": "object"},
            handler=lambda _args: {"ok": True},
        )
    )
    with pytest.raises(ValueError, match="tool name collision"):
        registry.register(
            ToolDefinition(
                name="lookup",
                description="Second",
                input_schema={"type": "object"},
                handler=lambda _args: {"ok": False},
            )
        )
