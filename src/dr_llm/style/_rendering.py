from __future__ import annotations

from typing import Any

from marimo_utils.ui._rendering import (
    auto_render as _auto_render,
    html_block as _html_block,
)


def auto_render(item: Any) -> Any:
    """Render a UI item through the marimo_utils compatibility layer."""
    return _auto_render(item)


def html_block(content: Any) -> Any:
    """Wrap rendered content in a block HTML container."""
    return _html_block(content)
