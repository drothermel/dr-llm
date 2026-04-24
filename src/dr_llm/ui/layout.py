from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import marimo as mo

from dr_llm.ui._rendering import auto_render


def wrap_cards(cards: Sequence[Any]) -> mo.Html:
    rendered = [auto_render(card) for card in cards if card is not None]
    return mo.hstack(rendered, wrap=True, justify="start", gap=1)


__all__ = ["wrap_cards"]
