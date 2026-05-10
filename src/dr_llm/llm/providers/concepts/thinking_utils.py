from __future__ import annotations


def matches_family(*, normalized: str, families: list[str]) -> bool:
    return any(
        normalized == family
        or is_snapshot_of_family(normalized=normalized, family=family)
        for family in families
    )


def is_snapshot_of_family(*, normalized: str, family: str) -> bool:
    prefix = f"{family}-"
    if not normalized.startswith(prefix):
        return False
    suffix = normalized[len(prefix) :]
    return bool(suffix) and suffix[0].isdigit()
