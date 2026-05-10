"""Rich renderers for pool summaries."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from itertools import islice
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.table import Table
from rich.text import Text

from dr_llm.pool.completion_filter import CompletionFilter
from dr_llm.pool.db.key_filter import PoolKeyFilter
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.pool_store import PoolStore

type PoolSummaryJustify = Literal["left", "center", "right"]

_WHITESPACE_RE = re.compile(r"\s+")


class PoolSummaryColumn(BaseModel):
    """Caller-defined column for a pool summary table."""

    model_config = ConfigDict(frozen=True)

    header: str
    value: Callable[[PoolSample], object]
    justify: PoolSummaryJustify = "left"
    style: str | None = None
    max_width: int | None = Field(default=None, ge=1)


def pool_summary_table(
    store: PoolStore,
    *,
    key_filter: PoolKeyFilter | None = None,
    completion: CompletionFilter = "all",
    limit: int | None = None,
    title: str | None = None,
    include_sample_id: bool = False,
    response_preview: Callable[[PoolSample], str | None] | None = None,
    response_max_chars: int = 80,
    extra_columns: Sequence[PoolSummaryColumn] = (),
) -> Table:
    """Build a Rich table summarizing pool samples."""
    if limit is not None and limit < 0:
        raise ValueError("limit must be >= 0")
    if response_max_chars < 1:
        raise ValueError("response_max_chars must be >= 1")

    rows = _load_rows(
        store,
        key_filter=key_filter,
        completion=completion,
        limit=limit,
    )
    table = _make_table(
        store,
        title=title,
        include_sample_id=include_sample_id,
        extra_columns=extra_columns,
    )

    for sample in rows:
        table.add_row(
            *_sample_cells(
                store,
                sample,
                include_sample_id=include_sample_id,
                response_preview=response_preview,
                response_max_chars=response_max_chars,
                extra_columns=extra_columns,
            )
        )

    table.caption = _caption(store, key_filter=key_filter, loaded=len(rows))
    return table


def print_pool_summary(
    store: PoolStore,
    *,
    console: Console | None = None,
    key_filter: PoolKeyFilter | None = None,
    completion: CompletionFilter = "all",
    limit: int | None = None,
    title: str | None = None,
    include_sample_id: bool = False,
    response_preview: Callable[[PoolSample], str | None] | None = None,
    response_max_chars: int = 80,
    extra_columns: Sequence[PoolSummaryColumn] = (),
) -> None:
    """Print a Rich table summarizing pool samples."""
    output = console or Console()
    output.print(
        pool_summary_table(
            store,
            key_filter=key_filter,
            completion=completion,
            limit=limit,
            title=title,
            include_sample_id=include_sample_id,
            response_preview=response_preview,
            response_max_chars=response_max_chars,
            extra_columns=extra_columns,
        )
    )


def _load_rows(
    store: PoolStore,
    *,
    key_filter: PoolKeyFilter | None,
    completion: CompletionFilter,
    limit: int | None,
) -> list[PoolSample]:
    samples = store.iter_samples(
        key_filter=key_filter,
        completion=completion,
    )
    if limit is None:
        return list(samples)
    return list(islice(samples, limit))


def _make_table(
    store: PoolStore,
    *,
    title: str | None,
    include_sample_id: bool,
    extra_columns: Sequence[PoolSummaryColumn],
) -> Table:
    table = Table(title=title or f"Pool Summary: {store.schema.name}")
    for column in store.schema.key_columns:
        table.add_column(column.name, overflow="fold")
    table.add_column("Idx", justify="right")
    table.add_column("Status")
    table.add_column("Attempts", justify="right")
    table.add_column("Finish")
    if include_sample_id:
        table.add_column("Sample ID", overflow="fold")
    for column in extra_columns:
        table.add_column(
            column.header,
            justify=column.justify,
            style=column.style,
            max_width=column.max_width,
            overflow="fold",
        )
    table.add_column("Response", overflow="fold", max_width=80)
    return table


def _sample_cells(
    store: PoolStore,
    sample: PoolSample,
    *,
    include_sample_id: bool,
    response_preview: Callable[[PoolSample], str | None] | None,
    response_max_chars: int,
    extra_columns: Sequence[PoolSummaryColumn],
) -> list[Text]:
    cells = [
        _format_cell(sample.key_values.get(column.name))
        for column in store.schema.key_columns
    ]
    cells.extend(
        [
            _format_cell(sample.sample_idx),
            _status_text(sample),
            _format_cell(sample.attempt_count),
            _format_cell(sample.finish_reason),
        ]
    )
    if include_sample_id:
        cells.append(_format_cell(sample.sample_id))
    cells.extend(
        _format_cell(column.value(sample)) for column in extra_columns
    )
    cells.append(
        _format_cell(
            _response_preview(sample, response_preview),
            max_chars=response_max_chars,
        )
    )
    return cells


def _caption(
    store: PoolStore,
    *,
    key_filter: PoolKeyFilter | None,
    loaded: int,
) -> str:
    progress = store.progress(key_filter=key_filter)
    return (
        f"Loaded {loaded:,} rows. "
        f"Total {progress.total:,}; "
        f"complete {progress.complete:,}; "
        f"incomplete {progress.incomplete:,}; "
        f"leased {progress.leased:,}; "
        f"errors {progress.error:,}."
    )


def _status_text(sample: PoolSample) -> Text:
    if sample.response is None:
        return Text("incomplete", style="yellow")
    if sample.finish_reason == "error":
        return Text("error", style="red")
    return Text("complete", style="green")


def _response_preview(
    sample: PoolSample,
    response_preview: Callable[[PoolSample], str | None] | None,
) -> str:
    if response_preview is not None:
        return response_preview(sample) or ""
    if sample.response is None:
        return ""
    text = sample.response.get("text")
    if isinstance(text, str):
        return text
    return _compact_json(sample.response)


def _format_cell(value: object, *, max_chars: int | None = None) -> Text:
    if value is None:
        rendered = ""
    elif isinstance(value, str):
        rendered = value
    elif isinstance(value, int | float | bool):
        rendered = str(value)
    else:
        rendered = _compact_json(value)

    normalized = _collapse_whitespace(rendered)
    if max_chars is not None:
        normalized = _truncate(normalized, max_chars)
    return Text(normalized)


def _compact_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), default=str
    )


def _collapse_whitespace(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value).strip()


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return "." * max_chars
    return f"{value[: max_chars - 3]}..."


__all__ = [
    "PoolSummaryColumn",
    "pool_summary_table",
    "print_pool_summary",
]
