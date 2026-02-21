import asyncio

from llm_pool.tools.executor import ToolExecutor
from llm_pool.tools.registry import ToolDefinition, ToolRegistry
from llm_pool.types import ProviderToolSpec, ToolInvocation


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
    assert result.error["error_type"] == "unknown_tool"


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
    assert result.error["error_type"] == "ToolExecutionError"


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
