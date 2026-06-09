from __future__ import annotations

from dr_llm.backends.fingerprint import fingerprint_request
from tests.backends._helpers import make_backend_request


def test_fingerprint_is_stable_for_same_logical_request() -> None:
    first = fingerprint_request(make_backend_request())
    second = fingerprint_request(make_backend_request())
    assert first == second
    assert len(first) == 64


def test_fingerprint_excludes_metadata_and_extensions() -> None:
    plain = fingerprint_request(make_backend_request())
    with_metadata = fingerprint_request(
        make_backend_request(
            metadata={"trace_id": "abc"},
            extensions={"tools": []},
        )
    )
    assert plain == with_metadata


def test_fingerprint_changes_when_generation_fields_change() -> None:
    base = fingerprint_request(make_backend_request())
    different_model = fingerprint_request(make_backend_request(model="gpt-4.1"))
    assert base != different_model
