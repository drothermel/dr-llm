from __future__ import annotations

import pytest

from dr_llm.errors import ProviderTransportError
from dr_llm.llm.providers.response_validation import validate_http_response


def test_validate_http_response_treats_408_as_transient() -> None:
    with pytest.raises(
        ProviderTransportError,
        match="demo transient error status=408 body=timeout",
    ):
        validate_http_response(
            provider_label="demo",
            status_code=408,
            response_text_preview="timeout",
            json_error=None,
            response_shape_error=None,
        )
