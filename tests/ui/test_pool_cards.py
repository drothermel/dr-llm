from __future__ import annotations

import marimo as mo
from dr_widget.inline import ActiveHtml

from dr_llm.pool.db.schema import KeyColumn, PoolSchema
from dr_llm.pool.models import PoolInspection, PoolInspectionStatus
from dr_llm.pool.pending.pending_status import PendingStatusCounts
from dr_llm.ui import PieChart, PoolSimpleStatsPieCard


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


def test_pool_simple_stats_pie_card_renders_as_marimo_html() -> None:
    card = PoolSimpleStatsPieCard(pool=demo_pool())

    rendered = card.render()

    assert isinstance(rendered, (mo.Html, ActiveHtml))


def test_pool_simple_stats_pie_card_total_matches_disjoint_card_buckets() -> None:
    card = PoolSimpleStatsPieCard(pool=demo_pool())

    assert card.total_samples == 1327


def test_pool_simple_stats_pie_card_uses_pie_chart_content() -> None:
    card = PoolSimpleStatsPieCard(pool=demo_pool())

    assert isinstance(card.pie_chart(), PieChart)
