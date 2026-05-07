from __future__ import annotations

from typing import Any

import pandas as pd


def single_df_row(frame: pd.DataFrame, *, source: str) -> dict[str, Any]:
    """Return the only row in a dataframe produced by a nested model."""
    row_count = len(frame.index)
    if row_count != 1:
        raise ValueError(
            f"{source}.to_df() must produce exactly one row; got {row_count}"
        )
    return dict(frame.iloc[0].to_dict())
