from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable
from typing import Any

from dr_llm.tools.registry import ToolRegistry
from dr_llm.types import ToolError, ToolErrorCode, ToolInvocation, ToolResult


async def _await_result(awaitable: Awaitable[Any]) -> Any:
    return await awaitable


class ToolExecutor:
    def __init__(self, *, registry: ToolRegistry) -> None:
        self.registry = registry

    @staticmethod
    def _error_result(
        *,
        tool_call_id: str,
        error_code: ToolErrorCode,
        exc: Exception,
    ) -> ToolResult:
        return ToolResult(
            tool_call_id=tool_call_id,
            ok=False,
            error=ToolError(
                error_code=error_code,
                message=str(exc),
                exception_type=type(exc).__name__,
            ),
        )

    @staticmethod
    def _ok_result(*, tool_call_id: str, result: Any) -> ToolResult:
        if not isinstance(result, dict):
            result = {"value": result}
        return ToolResult(tool_call_id=tool_call_id, ok=True, result=result)

    def invoke(self, call: ToolInvocation) -> ToolResult:
        try:
            tool = self.registry.get(call.name)
        except KeyError as exc:
            return self._error_result(
                tool_call_id=call.tool_call_id,
                error_code=ToolErrorCode.unknown_tool,
                exc=exc,
            )

        try:
            if inspect.iscoroutinefunction(tool.handler):
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    result = asyncio.run(tool.handler(call.arguments))
                else:
                    return self._error_result(
                        tool_call_id=call.tool_call_id,
                        error_code=ToolErrorCode.tool_async_in_running_loop,
                        exc=RuntimeError(
                            "Cannot run async tool handler synchronously while loop is running."
                        ),
                    )
            else:
                result_obj = tool.handler(call.arguments)
                if inspect.isawaitable(result_obj):
                    try:
                        asyncio.get_running_loop()
                    except RuntimeError:
                        result = asyncio.run(_await_result(result_obj))
                    else:
                        return self._error_result(
                            tool_call_id=call.tool_call_id,
                            error_code=ToolErrorCode.tool_async_in_running_loop,
                            exc=RuntimeError(
                                "Cannot await tool handler result synchronously while loop is running."
                            ),
                        )
                else:
                    result = result_obj
            return self._ok_result(tool_call_id=call.tool_call_id, result=result)
        except Exception as exc:  # noqa: BLE001
            return self._error_result(
                tool_call_id=call.tool_call_id,
                error_code=ToolErrorCode.tool_execution_failed,
                exc=exc,
            )

    async def invoke_async(self, call: ToolInvocation) -> ToolResult:
        try:
            tool = self.registry.get(call.name)
        except KeyError as exc:
            return self._error_result(
                tool_call_id=call.tool_call_id,
                error_code=ToolErrorCode.unknown_tool,
                exc=exc,
            )

        try:
            if inspect.iscoroutinefunction(tool.handler):
                result = await tool.handler(call.arguments)
            else:
                result_obj = tool.handler(call.arguments)
                result = (
                    await _await_result(result_obj)
                    if inspect.isawaitable(result_obj)
                    else result_obj
                )
            return self._ok_result(tool_call_id=call.tool_call_id, result=result)
        except Exception as exc:  # noqa: BLE001
            return self._error_result(
                tool_call_id=call.tool_call_id,
                error_code=ToolErrorCode.tool_execution_failed,
                exc=exc,
            )
