from __future__ import annotations

from collections.abc import Awaitable
from collections.abc import Callable
from threading import RLock
from typing import Any

from pydantic import BaseModel, ConfigDict

from llm_pool.types import ProviderToolSpec, ToolFunctionSpec


ToolHandler = Callable[[dict[str, Any]], dict[str, Any] | Awaitable[dict[str, Any]]]


class ToolDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler

    def to_provider_tool(self) -> ProviderToolSpec:
        return ProviderToolSpec(
            function=ToolFunctionSpec(
                name=self.name,
                description=self.description,
                parameters=self.input_schema,
            )
        )


class ToolRegistry:
    def __init__(self) -> None:
        self._lock = RLock()
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        key = tool.name.strip().lower()
        if not key:
            raise ValueError("tool.name must be non-empty")
        if tool.name != tool.name.strip():
            raise ValueError("tool.name must not have leading or trailing whitespace")
        with self._lock:
            existing = self._tools.get(key)
            if existing is not None and existing.name != tool.name:
                raise ValueError(
                    f"tool name collision for normalized key {key!r}: {existing.name!r} vs {tool.name!r}"
                )
            self._tools[key] = tool

    def get(self, name: str) -> ToolDefinition:
        key = name.strip().lower()
        with self._lock:
            tool = self._tools.get(key)
            known = ", ".join(sorted(self._tools.keys()))
        if tool is None:
            raise KeyError(f"Unknown tool {name!r}. Known tools: {known}")
        return tool

    def names(self) -> set[str]:
        with self._lock:
            return set(self._tools.keys())

    def to_provider_tools(self) -> list[ProviderToolSpec]:
        with self._lock:
            items = list(self._tools.values())
        return [tool.to_provider_tool() for tool in items]
