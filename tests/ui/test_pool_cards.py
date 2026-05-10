from __future__ import annotations

import marimo as mo
from dr_widget.inline import ActiveHtml

from dr_llm.pool.db.schema import KeyColumn, PoolSchema
from dr_llm.pool.admin.inspection import PoolInspection
from dr_llm.pool.pool_progress import PoolProgress
from dr_llm.ui import PieChart, PoolSimpleStatsPieCard


def demo_pool() -> PoolInspection:
    return PoolInspection(
        project_name="demo_project",
        name="demo_pool",
        pool_schema=PoolSchema(
            name="demo_pool",
            key_columns=[KeyColumn(name="provider"), KeyColumn(name="model")],
        ),
        progress=PoolProgress(
            total=1327, incomplete=44, leased=8, complete=1283, error=3
        ),
    )


def test_pool_simple_stats_pie_card_renders_as_marimo_html() -> None:
    card = PoolSimpleStatsPieCard(pool=demo_pool())

    rendered = card.render()

    assert isinstance(rendered, (mo.Html, ActiveHtml))


def test_pool_simple_stats_pie_card_total_matches_disjoint_card_buckets() -> (
    None
):
    card = PoolSimpleStatsPieCard(pool=demo_pool())

    assert card.total_samples == 1327


def test_pool_simple_stats_pie_card_uses_pie_chart_content() -> None:
    card = PoolSimpleStatsPieCard(pool=demo_pool())

    assert isinstance(card.pie_chart(), PieChart)


def test_pool_simple_stats_pie_card_uses_progress_buckets() -> None:
    card = PoolSimpleStatsPieCard(pool=demo_pool())

    values_by_label = {
        slice_.label.lower(): slice_.value for slice_ in card.pie_slices()
    }

    assert values_by_label == {
        "complete": 1280,
        "incomplete": 36,
        "leased": 8,
        "error": 3,
    }
