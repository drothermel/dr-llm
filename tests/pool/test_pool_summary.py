from __future__ import annotations

from collections.abc import Iterator
from io import StringIO
from typing import cast

from rich.console import Console
from rich.table import Table

from dr_llm.pool import (
    KeyColumn,
    PoolProgress,
    PoolSample,
    PoolSchema,
    PoolStore,
    PoolSummaryColumn,
    pool_summary_table,
    print_pool_summary,
)
from dr_llm.pool.completion_filter import CompletionFilter
from dr_llm.pool.db.key_filter import PoolKeyFilter


class StubPoolStore:
    def __init__(
        self,
        *,
        schema: PoolSchema,
        samples: list[PoolSample],
    ) -> None:
        self.schema = schema
        self.samples = samples
        self.iter_calls: list[CompletionFilter] = []

    def iter_samples(
        self,
        *,
        key_filter: PoolKeyFilter | None = None,
        completion: CompletionFilter = "all",
        chunk_size: int = 1000,
    ) -> Iterator[PoolSample]:
        _ = key_filter, chunk_size
        self.iter_calls.append(completion)
        return iter(self.samples)

    def progress(
        self,
        *,
        key_filter: PoolKeyFilter | None = None,
    ) -> PoolProgress:
        _ = key_filter
        complete = sum(sample.response is not None for sample in self.samples)
        error = sum(sample.finish_reason == "error" for sample in self.samples)
        return PoolProgress(
            total=len(self.samples),
            incomplete=len(self.samples) - complete,
            leased=0,
            complete=complete,
            error=error,
        )


def test_pool_summary_table_uses_schema_columns_and_extras() -> None:
    store = StubPoolStore(
        schema=_schema(),
        samples=[_sample(response={"text": "ok"}, metadata={"tokens": 7})],
    )

    table = pool_summary_table(
        cast(PoolStore, store),
        include_sample_id=True,
        extra_columns=[
            PoolSummaryColumn(
                header="Tokens",
                value=lambda sample: sample.metadata["tokens"],
                justify="right",
                max_width=6,
            )
        ],
    )

    assert [column.header for column in table.columns] == [
        "provider",
        "model",
        "Idx",
        "Status",
        "Attempts",
        "Finish",
        "Sample ID",
        "Tokens",
        "Response",
    ]
    assert table.columns[7].justify == "right"
    assert table.columns[7].max_width == 6


def test_pool_summary_table_renders_statuses_and_caption() -> None:
    store = StubPoolStore(
        schema=_schema(),
        samples=[
            _sample(response={"text": "done"}, finish_reason="stop"),
            _sample(response=None),
            _sample(response={"text": "failed"}, finish_reason="error"),
        ],
    )

    output = _render(pool_summary_table(cast(PoolStore, store)))

    assert "complete" in output
    assert "incomplete" in output
    assert "error" in output
    assert "Loaded 3 rows" in output
    assert "complete 2" in output
    assert "incomplete 1" in output
    assert "errors 1" in output


def test_pool_summary_table_previews_response_text_and_json() -> None:
    store = StubPoolStore(
        schema=_schema(),
        samples=[
            _sample(response={"text": "first\nsecond"}),
            _sample(response={"other": ["value"]}),
            _sample(response={"text": "0123456789abcdefghijklmnop"}),
        ],
    )

    output = _render(
        pool_summary_table(cast(PoolStore, store), response_max_chars=20)
    )

    assert "first second" in output
    assert '{"other":["value"]}' in output
    assert "0123456789abcdefg..." in output


def test_pool_summary_table_uses_limit_with_streamed_samples() -> None:
    store = StubPoolStore(
        schema=_schema(),
        samples=[
            _sample(provider="a"),
            _sample(provider="b"),
            _sample(provider="c"),
        ],
    )

    table = pool_summary_table(
        cast(PoolStore, store),
        completion="complete",
        limit=2,
    )

    assert len(table.rows) == 2
    assert store.iter_calls == ["complete"]


def test_print_pool_summary_uses_injected_console() -> None:
    output = StringIO()
    console = Console(
        file=output,
        force_terminal=False,
        width=140,
        color_system=None,
    )
    store = StubPoolStore(
        schema=_schema(),
        samples=[_sample(response={"text": "hello"})],
    )

    print_pool_summary(cast(PoolStore, store), console=console)

    assert "Pool Summary: test_pool" in output.getvalue()
    assert "hello" in output.getvalue()


def _schema() -> PoolSchema:
    return PoolSchema(
        name="test_pool",
        key_columns=[KeyColumn(name="provider"), KeyColumn(name="model")],
    )


def _sample(
    *,
    provider: str = "openai",
    model: str = "gpt-test",
    response: dict[str, object] | None = None,
    finish_reason: str | None = None,
    metadata: dict[str, object] | None = None,
) -> PoolSample:
    return PoolSample(
        sample_id=f"{provider}-{model}",
        key_values={"provider": provider, "model": model},
        sample_idx=1,
        response=response,
        finish_reason=finish_reason,
        metadata=metadata or {},
    )


def _render(table: Table) -> str:
    output = StringIO()
    console = Console(
        file=output,
        force_terminal=False,
        width=160,
        color_system=None,
    )
    console.print(table)
    return output.getvalue()
