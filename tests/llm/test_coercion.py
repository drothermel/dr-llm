from __future__ import annotations

from dr_llm.llm.coercion import as_int


def test_as_int_rejects_boolean_inputs() -> None:
    assert as_int(True) is None
    assert as_int(False) is None


def test_as_int_still_accepts_non_boolean_integer_values() -> None:
    assert as_int(1) == 1
    assert as_int("2") == 2
