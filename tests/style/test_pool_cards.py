from __future__ import annotations

import altair as alt
import marimo as mo

from dr_llm.pool.db.schema import KeyColumn, PoolSchema
from dr_llm.pool.models import PoolInspection
from dr_llm.pool.models import PoolInspectionStatus
from dr_llm.pool.pending.pending_status import PendingStatusCounts
from dr_llm.style import ColorPalette, PiePoolCard, PoolCard


def demo_pool() -> PoolInspection:
    return PoolInspection(
        project_name="demo_project",
        name="demo_pool",
        pool_schema=PoolSchema(
            name="demo_pool",
            key_columns=[KeyColumn(name="provider"), KeyColumn(name="model")],
        ),
        sample_count=1280,
        pending_counts=PendingStatusCounts(pending=36, leased=8, failed=3),
        status=PoolInspectionStatus.in_progress,
    )


def test_pool_card_renders_as_marimo_html() -> None:
    card = PoolCard(pool=demo_pool(), palette=ColorPalette.default())

    rendered = card.render()

    assert isinstance(rendered, mo.Html)


def test_pie_pool_card_total_matches_disjoint_card_buckets() -> None:
    card = PiePoolCard(pool=demo_pool(), palette=ColorPalette.default())

    assert card.total_samples == 1327


def test_pie_pool_card_uses_altair_chart_content() -> None:
    card = PiePoolCard(pool=demo_pool(), palette=ColorPalette.default())

    rendered_chart = card.pie_chart()

    assert isinstance(rendered_chart, alt.LayerChart)
