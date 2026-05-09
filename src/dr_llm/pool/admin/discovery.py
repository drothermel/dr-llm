from __future__ import annotations

import re
from typing import Final

from sqlalchemy import text

from dr_llm.pool.db import DbConfig, DbRuntime, PoolTableType

POOL_TABLE_RE = re.compile(rf"^pool_(.+)_{re.escape(PoolTableType.SAMPLES)}$")
POOL_DISCOVERY_SQL = text(
    "SELECT table_name FROM information_schema.tables "
    "WHERE table_schema = 'public' "
    rf"AND table_name LIKE 'pool\_%\_{PoolTableType.SAMPLES}' "
    "ORDER BY table_name"
)
_DEFAULT_TESTISH_TOKENS: Final[frozenset[str]] = frozenset(
    {"test", "tst", "smoke", "demo"}
)


def discover_pools(dsn: str) -> list[str]:
    runtime = DbRuntime(DbConfig(dsn=dsn))
    try:
        return discover_pools_from_runtime(runtime)
    finally:
        runtime.close()


def discover_pools_from_runtime(runtime: DbRuntime) -> list[str]:
    with runtime.connect() as conn:
        rows = conn.execute(POOL_DISCOVERY_SQL).fetchall()
    return [
        match.group(1)
        for (table_name,) in rows
        if (match := POOL_TABLE_RE.match(table_name))
    ]


def pool_name_tokens(pool_name: str) -> list[str]:
    return [token.lower() for token in pool_name.split("_") if token]


def pool_name_has_token_match(
    pool_name: str,
    match_tokens: list[str] | None = None,
) -> bool:
    token_set = set(match_tokens or _DEFAULT_TESTISH_TOKENS)
    return any(token in token_set for token in pool_name_tokens(pool_name))
