from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from datetime import datetime
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit, urlunsplit

import psycopg
from pydantic import BaseModel, ConfigDict
from psycopg.conninfo import conninfo_to_dict
from psycopg import sql

from dr_llm.datetime_utils import UTC
from dr_llm.project.docker_psql import (
    call_docker_pg_dump_stream,
    validate_pg_identifier,
)
from dr_llm.project.errors import ProjectError
from dr_llm.project.models import (
    ProjectPostgresSyncResult,
    ProjectSyncValidation,
)
from dr_llm.project.project_info import ProjectInfo
from dr_llm.project.project_service import get_project

logger = logging.getLogger(__name__)


class SyncPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    project_name: str
    source_dsn: str
    admin_url: str
    target_database: str
    temporary_database: str
    previous_database: str
    temporary_dsn: str
    drop_previous: bool


class PsqlRestoreTarget(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dbname: str
    user: str
    password: str = ""
    host: str = "localhost"
    port: str = "5432"
    sslmode: str | None = None


class DatabaseSwapPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    admin_url: str
    temporary_database: str
    target_database: str
    previous_database: str


def sync_project_to_postgres(
    name: str,
    admin_url: str,
    *,
    target_database: str | None = None,
    drop_previous: bool = False,
) -> ProjectPostgresSyncResult:
    """Replace a Postgres database with a validated local project snapshot."""

    plan = _build_sync_plan(
        name=name,
        admin_url=admin_url,
        target_database=target_database,
        drop_previous=drop_previous,
    )
    started_at = perf_counter()
    logger.info(
        "Starting Postgres sync for project %r to database %r",
        plan.project_name,
        plan.target_database,
    )
    try:
        _restore_temporary_database(plan)
        validation = _validate_temporary_database(plan)
        replaced_existing = _swap_validated_database(plan)
        _drop_previous_database_if_requested(plan, replaced_existing)
        logger.info(
            "Completed Postgres sync for project %r to database %r in %.2fs",
            plan.project_name,
            plan.target_database,
            perf_counter() - started_at,
        )
        return _build_sync_result(plan, validation, replaced_existing)
    except Exception:
        with suppress(Exception):
            _drop_database(plan.admin_url, plan.temporary_database)
        raise


def _build_sync_plan(
    *,
    name: str,
    admin_url: str,
    target_database: str | None,
    drop_previous: bool,
) -> SyncPlan:
    project = get_project(name)
    if project.dsn is None:
        raise ProjectError(f"Project {name!r} has no DSN; start it first.")

    target_name = validate_pg_identifier(
        target_database or name, "database name"
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    temporary_database = _sync_database_name(target_name, "sync", timestamp)
    previous_database = _sync_database_name(target_name, "prev", timestamp)
    return SyncPlan(
        project_name=name,
        source_dsn=project.dsn,
        admin_url=admin_url,
        target_database=target_name,
        temporary_database=temporary_database,
        previous_database=previous_database,
        temporary_dsn=_url_for_database(admin_url, temporary_database),
        drop_previous=drop_previous,
    )


def _restore_temporary_database(plan: SyncPlan) -> None:
    _create_database(plan.admin_url, plan.temporary_database)
    with tempfile.NamedTemporaryFile(
        prefix=f"dr_llm_{plan.project_name}_",
        suffix=".sql",
        delete=True,
    ) as dump_file:
        dump_path = Path(dump_file.name)
        _dump_project_to_file(plan.project_name, dump_path)
        _restore_sql_file(plan.temporary_dsn, dump_path)


def _validate_temporary_database(plan: SyncPlan) -> ProjectSyncValidation:
    validation = validate_project_database_copy(
        source_dsn=plan.source_dsn,
        target_dsn=plan.temporary_dsn,
    )
    if not validation.passed:
        raise ProjectError(
            "Postgres sync validation failed before swap: "
            + "; ".join(validation.mismatches)
        )
    return validation


def _swap_validated_database(plan: SyncPlan) -> bool:
    return _replace_database(
        admin_url=plan.admin_url,
        temporary_database=plan.temporary_database,
        target_database=plan.target_database,
        previous_database=plan.previous_database,
    )


def _drop_previous_database_if_requested(
    plan: SyncPlan, replaced_existing: bool
) -> None:
    if plan.drop_previous and replaced_existing:
        _drop_database(plan.admin_url, plan.previous_database)


def _build_sync_result(
    plan: SyncPlan,
    validation: ProjectSyncValidation,
    replaced_existing: bool,
) -> ProjectPostgresSyncResult:
    return ProjectPostgresSyncResult(
        project_name=plan.project_name,
        target_database=plan.target_database,
        temporary_database=plan.temporary_database,
        previous_database=plan.previous_database
        if replaced_existing
        else None,
        previous_database_dropped=plan.drop_previous and replaced_existing,
        validation=validation,
    )


def validate_project_database_copy(
    *,
    source_dsn: str,
    target_dsn: str,
) -> ProjectSyncValidation:
    with psycopg.connect(source_dsn) as source_conn:
        with psycopg.connect(target_dsn) as target_conn:
            return _validate_project_database_copy(
                source_conn=source_conn,
                target_conn=target_conn,
            )


def _validate_project_database_copy(
    *,
    source_conn: psycopg.Connection[tuple],
    target_conn: psycopg.Connection[tuple],
) -> ProjectSyncValidation:
    source_tables = _public_table_names(source_conn)
    target_tables = _public_table_names(target_conn)
    mismatches: list[str] = []
    if source_tables != target_tables:
        missing = sorted(set(source_tables) - set(target_tables))
        extra = sorted(set(target_tables) - set(source_tables))
        if missing:
            mismatches.append("missing target tables: " + ", ".join(missing))
        if extra:
            mismatches.append("extra target tables: " + ", ".join(extra))

    source_pool_count = _pool_catalog_count(source_conn)
    target_pool_count = _pool_catalog_count(target_conn)
    if source_pool_count != target_pool_count:
        mismatches.append(
            "pool_catalog count mismatch: "
            f"source={source_pool_count} target={target_pool_count}"
        )

    checked_table_count = 0
    if source_tables == target_tables:
        for table_name in source_tables:
            source_rows = _table_row_count(source_conn, table_name)
            target_rows = _table_row_count(target_conn, table_name)
            checked_table_count += 1
            if source_rows != target_rows:
                mismatches.append(
                    f"{table_name} row count mismatch: "
                    f"source={source_rows} target={target_rows}"
                )

    return ProjectSyncValidation(
        source_table_count=len(source_tables),
        target_table_count=len(target_tables),
        source_pool_count=source_pool_count,
        target_pool_count=target_pool_count,
        checked_table_count=checked_table_count,
        mismatches=mismatches,
    )


def _sync_database_name(
    target_database: str, label: str, timestamp: str
) -> str:
    suffix = f"_{label}_{timestamp}"
    max_prefix_length = max(0, 63 - len(suffix))
    return validate_pg_identifier(
        f"{target_database[:max_prefix_length]}{suffix}", "database name"
    )


def _url_for_database(base_url: str, database_name: str) -> str:
    parts = urlsplit(base_url)
    if not parts.scheme or not parts.netloc:
        raise ProjectError("Database URL must include a scheme and host.")
    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            f"/{database_name}",
            parts.query,
            parts.fragment,
        )
    )


def _dump_project_to_file(name: str, dump_path: Path) -> None:
    with dump_path.open("wb") as output_stream:
        call_docker_pg_dump_stream(
            container_name=ProjectInfo.container_name_for(name),
            db_user=ProjectInfo.db_user,
            db_name=ProjectInfo.db_name,
            output_stream=output_stream,
            extra_args=("--no-owner", "--no-privileges"),
        )


def _restore_sql_file(target_dsn: str, dump_path: Path) -> None:
    target = _parse_restore_target(target_dsn)
    with _temporary_pgpass_file(target) as pgpass_path:
        result = _run_psql_restore(
            target=target,
            dump_path=dump_path,
            pgpass_path=pgpass_path,
        )
    _raise_for_psql_restore_failure(result)


def _parse_restore_target(target_dsn: str) -> PsqlRestoreTarget:
    conninfo = conninfo_to_dict(target_dsn)
    dbname_value = conninfo.get("dbname")
    user_value = conninfo.get("user")
    if dbname_value is None or user_value is None:
        raise ProjectError("Target DSN must include a database and user.")
    sslmode = conninfo.get("sslmode")
    return PsqlRestoreTarget(
        dbname=str(dbname_value),
        user=str(user_value),
        password=str(conninfo.get("password", "")),
        host=str(conninfo.get("host", "localhost")),
        port=str(conninfo.get("port", "5432")),
        sslmode=str(sslmode) if sslmode is not None else None,
    )


@contextmanager
def _temporary_pgpass_file(target: PsqlRestoreTarget) -> Iterator[Path]:
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        prefix="dr_llm_pgpass_",
        delete=False,
    ) as pgpass:
        pgpass_path = Path(pgpass.name)
        os.chmod(pgpass_path, 0o600)
        pgpass.write(
            _pgpass_line(
                target.host,
                target.port,
                target.dbname,
                target.user,
                target.password,
            )
        )

    try:
        yield pgpass_path
    finally:
        pgpass_path.unlink(missing_ok=True)


def _run_psql_restore(
    *,
    target: PsqlRestoreTarget,
    dump_path: Path,
    pgpass_path: Path,
) -> subprocess.CompletedProcess[bytes]:
    try:
        with dump_path.open("rb") as sql_stream:
            return subprocess.run(
                _psql_restore_command(target),
                stdin=sql_stream,
                env=_psql_restore_env(target, pgpass_path),
                capture_output=True,
                check=False,
            )
    except FileNotFoundError as exc:
        raise ProjectError(
            "psql is required to restore remote databases."
        ) from exc


def _psql_restore_env(
    target: PsqlRestoreTarget, pgpass_path: Path
) -> Mapping[str, str]:
    env = os.environ.copy()
    env["PGPASSFILE"] = str(pgpass_path)
    if target.sslmode is not None:
        env["PGSSLMODE"] = target.sslmode
    return env


def _psql_restore_command(target: PsqlRestoreTarget) -> list[str]:
    return [
        "psql",
        "-h",
        target.host,
        "-p",
        target.port,
        "-U",
        target.user,
        "-d",
        target.dbname,
        "-v",
        "ON_ERROR_STOP=1",
        "-q",
    ]


def _raise_for_psql_restore_failure(
    result: subprocess.CompletedProcess[bytes],
) -> None:
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        raise ProjectError(f"psql restore failed: {stderr or 'unknown error'}")


def _pgpass_line(
    host: str, port: str, dbname: str, user: str, password: str
) -> str:
    fields = (host, port, dbname, user, password)
    return ":".join(_pgpass_field(field) for field in fields) + "\n"


def _pgpass_field(value: str) -> str:
    return value.replace("\\", "\\\\").replace(":", "\\:")


def _connect_admin(admin_url: str) -> psycopg.Connection[tuple]:
    try:
        return psycopg.connect(admin_url, autocommit=True)
    except psycopg.Error as exc:
        raise ProjectError(
            f"Could not connect to remote Postgres: {exc}"
        ) from exc


def _create_database(admin_url: str, database_name: str) -> None:
    with _connect_admin(admin_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("CREATE DATABASE {}").format(
                    sql.Identifier(database_name)
                )
            )


def _drop_database(admin_url: str, database_name: str) -> None:
    with _connect_admin(admin_url) as conn:
        with conn.cursor() as cur:
            _terminate_database_connections(cur, database_name)
            cur.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                    sql.Identifier(database_name)
                )
            )


def _replace_database(
    *,
    admin_url: str,
    temporary_database: str,
    target_database: str,
    previous_database: str,
) -> bool:
    return _replace_database_from_plan(
        DatabaseSwapPlan(
            admin_url=admin_url,
            temporary_database=temporary_database,
            target_database=target_database,
            previous_database=previous_database,
        )
    )


def _replace_database_from_plan(plan: DatabaseSwapPlan) -> bool:
    with _connect_admin(plan.admin_url) as conn:
        with conn.cursor() as cur:
            target_exists = _move_existing_target_to_previous(cur, plan)
            try:
                _rename_database(
                    cur, plan.temporary_database, plan.target_database
                )
            except Exception as exc:
                if not target_exists:
                    raise
                _restore_previous_target_after_failed_swap(cur, plan)
                raise exc
    return target_exists


def _move_existing_target_to_previous(
    cur: psycopg.Cursor[tuple],
    plan: DatabaseSwapPlan,
) -> bool:
    if not _database_exists(cur, plan.target_database):
        return False
    _terminate_database_connections(cur, plan.target_database)
    _rename_database(cur, plan.target_database, plan.previous_database)
    return True


def _restore_previous_target_after_failed_swap(
    cur: psycopg.Cursor[tuple],
    plan: DatabaseSwapPlan,
) -> None:
    logger.exception(
        "Failed to rename temporary database %r to %r; "
        "attempting to restore previous target %r.",
        plan.temporary_database,
        plan.target_database,
        plan.previous_database,
    )
    try:
        _terminate_database_connections(cur, plan.previous_database)
        _rename_database(cur, plan.previous_database, plan.target_database)
        logger.info(
            "Restored previous target database %r after failed swap.",
            plan.target_database,
        )
    except Exception as rollback_exc:
        _raise_swap_failure_after_rollback_failure(
            plan.target_database, rollback_exc
        )


def _raise_swap_failure_after_rollback_failure(
    target_database: str, rollback_exc: Exception
) -> None:
    logger.exception(
        "Failed to restore previous target database %r.", target_database
    )
    raise ProjectError(
        "Postgres sync database swap failed and rollback also failed."
    ) from rollback_exc


def _rename_database(
    cur: psycopg.Cursor[tuple],
    source_database: str,
    target_database: str,
) -> None:
    cur.execute(
        sql.SQL("ALTER DATABASE {} RENAME TO {}").format(
            sql.Identifier(source_database),
            sql.Identifier(target_database),
        )
    )


def _database_exists(
    cur: psycopg.Cursor[tuple],
    database_name: str,
) -> bool:
    cur.execute(
        "SELECT 1 FROM pg_database WHERE datname = %s",
        [database_name],
    )
    return cur.fetchone() is not None


def _terminate_database_connections(
    cur: psycopg.Cursor[tuple],
    database_name: str,
) -> None:
    cur.execute(
        """
        SELECT pg_terminate_backend(pid)
        FROM pg_stat_activity
        WHERE datname = %s
          AND pid <> pg_backend_pid()
        """,
        [database_name],
    )


def _public_table_names(conn: psycopg.Connection[tuple]) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT relname
            FROM pg_class
            WHERE relkind = 'r'
              AND relnamespace = 'public'::regnamespace
            ORDER BY relname
            """
        )
        return [row[0] for row in cur.fetchall()]


def _pool_catalog_count(conn: psycopg.Connection[tuple]) -> int | None:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.pool_catalog') IS NOT NULL")
        exists_row = cur.fetchone()
        assert exists_row is not None
        if not exists_row[0]:
            return None
        cur.execute("SELECT count(*) FROM pool_catalog")
        count_row = cur.fetchone()
        assert count_row is not None
        return count_row[0]


def _table_row_count(conn: psycopg.Connection[tuple], table_name: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT count(*) FROM {}").format(
                sql.Identifier(table_name)
            )
        )
        count_row = cur.fetchone()
        assert count_row is not None
        return count_row[0]
