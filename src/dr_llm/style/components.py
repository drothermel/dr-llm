from __future__ import annotations

from typing import ClassVar

import marimo as mo
from dr_widget.inline import ActiveHtml
from mohtml import div, span  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict

from dr_llm.pool.models import PendingStatusCounts
from dr_llm.style.theme import Style
from marimo_utils.ui._rendering import auto_render, html_block
from marimo_utils.ui import Badge, BadgeVariant, DataItem


class AxisBadge(BaseModel):
    model_config = ConfigDict(frozen=True)

    MATCHED_CLASSES: ClassVar[list[tuple[str, str]]] = [
        ("llm", "border-blue-300 bg-blue-50 text-blue-900"),
        ("prompt", "border-amber-300 bg-amber-50 text-amber-900"),
        ("data", "border-emerald-300 bg-emerald-50 text-emerald-900"),
    ]
    DEFAULT_CLASSES: ClassVar[str] = "border-slate-300 bg-slate-50 text-slate-900"

    label: str

    def badge_classes(self) -> str:
        normalized = self.label.lower()
        for needle, classes in self.MATCHED_CLASSES:
            if needle in normalized:
                return classes
        return self.DEFAULT_CLASSES

    def render(self) -> mo.Html | ActiveHtml:
        return Badge(
            label=self.label,
            variant=BadgeVariant.OUTLINE,
            klass=self.badge_classes(),
        ).render()


class AxesLabel(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    items: list[object]

    def render(self) -> mo.Html:
        rendered_items = [auto_render(item) for item in self.items]
        return html_block(
            div(
                span("Axes:", klass="text-sm font-semibold text-foreground"),
                *rendered_items,
                klass="inline-flex flex-wrap items-center gap-2",
            )
        )


class PendingDataItems(BaseModel):
    model_config = ConfigDict(frozen=True)

    pending_counts: PendingStatusCounts
    style: Style

    def items(self) -> list[DataItem]:
        pending_counts_model = type(self.pending_counts)
        return [
            DataItem(
                label=field_name.title(),
                value=str(getattr(self.pending_counts, field_name)),
            )
            for field_name in pending_counts_model.model_fields
        ]

    def render(self) -> mo.Html:
        return mo.vstack([item.render() for item in self.items()], gap=0)


__all__ = ["AxesLabel", "AxisBadge", "PendingDataItems"]
