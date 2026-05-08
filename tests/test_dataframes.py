from __future__ import annotations

import pandas as pd
import pytest

from dr_llm.dataframes import single_df_row


def test_single_df_row_returns_only_row() -> None:
    row = single_df_row(
        pd.DataFrame.from_records([{"name": "demo", "count": 3}]),
        source="DemoModel",
    )

    assert row == {"name": "demo", "count": 3}


def test_single_df_row_rejects_empty_frame() -> None:
    with pytest.raises(ValueError, match=r"DemoModel\.to_df\(\).*got 0"):
        single_df_row(pd.DataFrame(), source="DemoModel")


def test_single_df_row_rejects_multi_row_frame() -> None:
    frame = pd.DataFrame.from_records([{"name": "a"}, {"name": "b"}])

    with pytest.raises(ValueError, match=r"DemoModel\.to_df\(\).*got 2"):
        single_df_row(frame, source="DemoModel")
