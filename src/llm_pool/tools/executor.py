from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable
from typing import Any

from llm_pool.errors import ToolExecutionError
from llm_pool.tools.registry import ToolRegistry
from llm_pool.types import ToolInvocation, ToolResult


async def _await_result(awaitable: Awaitable[dict[str, Any]]) -> dict[str, Any]:
    return await awaitable


class ToolExecutor:
    def __init__(self, *, registry: ToolRegistry) -> None:
        self.registry = registry

    def invoke(self, call: ToolInvocation) -> ToolResult:
        try:
            tool = self.registry.get(call.name)
        except KeyError as exc:
            return ToolResult(
                tool_call_id=call.tool_call_id,
                ok=False,
                error={"error_type": "unknown_tool", "message": str(exc)},
            )

        try:
            if inspect.iscoroutinefunction(tool.handler):
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    result = asyncio.run(tool.handler(call.arguments))
                else:
                    raise ToolExecutionError(
                        "Cannot run async tool handler synchronously while loop is running."
                    ) from None
            else:
                result_obj = tool.handler(call.arguments)
                if inspect.isawaitable(result_obj):
                    try:
                        asyncio.get_running_loop()
                    except RuntimeError:
                        result = asyncio.run(_await_result(result_obj))
                    else:
                        raise ToolExecutionError(
                            "Cannot await tool handler result synchronously while loop is running."
                        ) from None
                else:
                    result = result_obj
            if not isinstance(result, dict):
                result = {"value": result}
            return ToolResult(tool_call_id=call.tool_call_id, ok=True, result=result)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                tool_call_id=call.tool_call_id,
                ok=False,
                error={"error_type": type(exc).__name__, "message": str(exc)},
            )
