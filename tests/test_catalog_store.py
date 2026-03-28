from __future__ import annotations

import pytest

from dr_llm.storage._catalog_store import _row_to_entry


def test_row_to_entry_rejects_unexpected_source_quality() -> None:
    with pytest.raises(ValueError, match="Unexpected source_quality"):
        _row_to_entry(
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "display_name": None,
                "context_window": None,
                "max_output_tokens": None,
                "supports_reasoning": None,
                "supports_tools": None,
                "supports_vision": None,
                "pricing_json": {},
                "rate_limits_json": {},
                "source_quality": "overlay",
                "metadata_json": {},
                "updated_at": None,
            }
        )
