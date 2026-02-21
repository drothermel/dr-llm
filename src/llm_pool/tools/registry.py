from __future__ import annotations

from collections.abc import Awaitable
from collections.abc import Callable
from threading import RLock
from typing import Any

from pydantic import BaseModel, ConfigDict


ToolHandler = Callable[[dict[str, Any]], dict[str, Any] | Awaitable[dict[str, Any]]]


class ToolDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler


class ToolRegistry:
    def __init__(self) -> None:
        self._lock = RLock()
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        key = tool.name.strip()
        if not key:
            raise ValueError("tool.name must be non-empty")
        with self._lock:
            self._tools[key] = tool

    def get(self, name: str) -> ToolDefinition:
        key = name.strip()
        with self._lock:
            tool = self._tools.get(key)
        if tool is None:
            known = ", ".join(sorted(self.names()))
            raise KeyError(f"Unknown tool {name!r}. Known tools: {known}")
        return tool

    def names(self) -> set[str]:
        with self._lock:
            return set(self._tools.keys())

    def to_provider_tools(self) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._tools.values())
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in items
        ]
