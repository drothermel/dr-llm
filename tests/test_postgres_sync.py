from __future__ import annotations

import subprocess
from pathlib import Path
from typing import BinaryIO, cast

import pytest
from pydantic import BaseModel, ConfigDict, Field

import dr_llm.project.postgres_sync as postgres_sync_module
from dr_llm.project.docker_project_metadata import ContainerStatus
from dr_llm.project.errors import ProjectError
from dr_llm.project.models import ProjectSyncValidation
from dr_llm.project.project_info import ProjectInfo


class ValidationCall(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_dsn: str
    target_dsn: str


class ProjectLookupFake(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    project: ProjectInfo

    def get_project(self, name: str) -> ProjectInfo:
        assert name == self.project.name
        return self.project


class SyncCallRecorder(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operations: list[str] = Field(default_factory=list)
    created_databases: list[str] = Field(default_factory=list)
    dumped_projects: list[str] = Field(default_factory=list)
    restored_dsns: list[str] = Field(default_factory=list)
    validations: list[ValidationCall] = Field(default_factory=list)
    replaced_databases: list[str] = Field(default_factory=list)
    dropped_databases: list[str] = Field(default_factory=list)
    validation_result: ProjectSyncValidation = Field(
        default_factory=lambda: ProjectSyncValidation(
            source_table_count=1,
            target_table_count=1,
            checked_table_count=1,
        )
    )
    replaced_existing: bool = True

    @property
    def operation_names(self) -> list[str]:
        return self.operations

    def create_database(self, admin_url: str, database_name: str) -> None:
        _ = admin_url
        self.operations.append("create")
        self.created_databases.append(database_name)

    def dump_project_to_file(self, project_name: str, dump_path: Path) -> None:
        self.operations.append("dump")
        self.dumped_projects.append(project_name)
        dump_path.write_text("select 1;\n")

    def restore_sql_file(self, target_dsn: str, dump_path: Path) -> None:
        _ = dump_path
        self.operations.append("restore")
        self.restored_dsns.append(target_dsn)

    def validate_project_database_copy(
        self, *, source_dsn: str, target_dsn: str
    ) -> ProjectSyncValidation:
        self.operations.append("validate")
        self.validations.append(
            ValidationCall(source_dsn=source_dsn, target_dsn=target_dsn)
        )
        return self.validation_result

    def replace_database(
        self, plan: postgres_sync_module.DatabaseSwapPlan
    ) -> bool:
        self.operations.append("replace")
        self.replaced_databases.append(plan.target_database)
        return self.replaced_existing

    def drop_database(self, admin_url: str, database_name: str) -> None:
        _ = admin_url
        self.operations.append("drop")
        self.dropped_databases.append(database_name)


def _running_project(name: str) -> ProjectInfo:
    return ProjectInfo(
        name=name,
        port=5500,
        status=ContainerStatus.RUNNING,
    )


def _sync_service(
    calls: SyncCallRecorder,
    *,
    project: ProjectInfo | None = None,
) -> postgres_sync_module.PostgresSyncService:
    return postgres_sync_module.PostgresSyncService(
        project_lookup=ProjectLookupFake(
            project=project or _running_project("demo")
        ),
        transfer=calls,
        validator=calls,
        admin=calls,
    )


def _successful_sync_calls() -> SyncCallRecorder:
    calls = SyncCallRecorder()
    return calls


def test_build_sync_plan_derives_temporary_database_target() -> None:
    calls = SyncCallRecorder()
    service = _sync_service(calls)

    plan = service.build_sync_plan(
        name="demo",
        admin_url="postgresql://example/postgres?sslmode=require",
        target_database="remote_demo",
        drop_previous=True,
    )

    assert plan.project_name == "demo"
    assert plan.source_dsn == (
        "postgresql://postgres:postgres@localhost:5500/dr_llm"
    )
    assert plan.target_database == "remote_demo"
    assert plan.temporary_database.startswith("remote_demo_sync_")
    assert plan.previous_database.startswith("remote_demo_prev_")
    assert plan.temporary_dsn == (
        f"postgresql://example/{plan.temporary_database}?sslmode=require"
    )
    assert plan.drop_previous is True


def test_sync_project_to_postgres_validates_and_swaps_database() -> None:
    calls = _successful_sync_calls()
    service = _sync_service(calls)

    result = service.sync_project_to_postgres(
        "demo",
        "postgresql://example/postgres?sslmode=require",
        drop_previous=True,
    )

    assert result.success is True
    assert result.target_database == "demo"
    assert result.previous_database is not None
    assert result.previous_database_dropped is True
    assert calls.operation_names == [
        "create",
        "dump",
        "restore",
        "validate",
        "replace",
        "drop",
    ]
    assert calls.dumped_projects == ["demo"]


def test_sync_project_to_postgres_restores_and_validates_temporary_database() -> (
    None
):
    calls = _successful_sync_calls()
    service = _sync_service(calls)

    result = service.sync_project_to_postgres(
        "demo",
        "postgresql://example/postgres?sslmode=require",
    )

    expected_target_dsn = (
        f"postgresql://example/{result.temporary_database}?sslmode=require"
    )
    assert calls.created_databases == [result.temporary_database]
    assert calls.restored_dsns == [expected_target_dsn]
    assert calls.validations == [
        ValidationCall(
            source_dsn="postgresql://postgres:postgres@localhost:5500/dr_llm",
            target_dsn=expected_target_dsn,
        )
    ]


def test_sync_project_to_postgres_drops_temp_database_when_validation_fails() -> (
    None
):
    calls = SyncCallRecorder(
        validation_result=ProjectSyncValidation(
            source_table_count=1,
            target_table_count=1,
            checked_table_count=1,
            mismatches=["alpha row count mismatch: source=1 target=0"],
        ),
    )
    service = _sync_service(calls)

    with pytest.raises(ProjectError, match="validation failed"):
        service.sync_project_to_postgres(
            "demo", "postgresql://example/postgres"
        )

    assert len(calls.dropped_databases) == 1
    assert "_sync_" in calls.dropped_databases[0]


def test_sync_project_to_postgres_requires_project_dsn() -> None:
    calls = SyncCallRecorder()
    service = _sync_service(
        calls,
        project=ProjectInfo(name="demo", status=ContainerStatus.STOPPED),
    )

    with pytest.raises(ProjectError, match="has no DSN"):
        service.sync_project_to_postgres(
            "demo", "postgresql://example/postgres"
        )


def test_sync_project_to_postgres_rejects_invalid_admin_url() -> None:
    calls = SyncCallRecorder()
    service = _sync_service(calls)

    with pytest.raises(ProjectError, match="scheme and host"):
        service.sync_project_to_postgres("demo", "postgres")

    assert calls.operation_names == []


def test_pgpass_line_escapes_colons_and_backslashes() -> None:
    line = postgres_sync_module._pgpass_line(
        "db:host",
        "5432",
        "db\\name",
        "user:name",
        r"pa:ss\word",
    )

    assert line == r"db\:host:5432:db\\name:user\:name:pa\:ss\\word" + "\n"


def test_parse_restore_target_reads_required_dsn_fields() -> None:
    target_dsn = (
        "postgresql://alice:secret@example.com:6543/demo"
        "?sslmode=require&channel_binding=require"
    )
    target = postgres_sync_module._parse_restore_target(target_dsn)

    assert target.dsn == target_dsn
    assert target.dbname == "demo"
    assert target.user == "alice"
    assert target.password == "secret"
    assert target.host == "example.com"
    assert target.port == "6543"


@pytest.mark.parametrize(
    "conninfo",
    [
        {"user": "alice"},
        {"dbname": "demo"},
    ],
)
def test_parse_restore_target_requires_database_and_user(
    monkeypatch: pytest.MonkeyPatch,
    conninfo: dict[str, str],
) -> None:
    monkeypatch.setattr(
        postgres_sync_module,
        "conninfo_to_dict",
        lambda _target_dsn: conninfo,
    )

    with pytest.raises(ProjectError, match="database and user"):
        postgres_sync_module._parse_restore_target("postgresql://example")


def test_restore_sql_file_uses_full_dsn_and_removes_pgpass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dump_path = tmp_path / "dump.sql"
    dump_path.write_bytes(b"select 1;\n")
    target_dsn = (
        "postgresql://alice:secret@example.com:6543/demo"
        "?sslmode=require&channel_binding=require"
    )
    captured: dict[str, object] = {}

    def fake_run(
        command: list[str],
        *,
        stdin: BinaryIO,
        env: dict[str, str],
        capture_output: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[bytes]:
        pgpass_path = Path(env["PGPASSFILE"])
        captured["command"] = command
        captured["stdin"] = stdin.read()
        captured["env"] = env
        captured["pgpass_path"] = pgpass_path
        captured["pgpass"] = pgpass_path.read_text()
        captured["capture_output"] = capture_output
        captured["check"] = check
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(postgres_sync_module.subprocess, "run", fake_run)

    postgres_sync_module._restore_sql_file(target_dsn, dump_path)

    assert captured["command"] == [
        "psql",
        target_dsn,
        "-v",
        "ON_ERROR_STOP=1",
        "-q",
    ]
    assert captured["stdin"] == b"select 1;\n"
    env = cast(dict[str, str], captured["env"])
    assert "PGSSLMODE" not in env
    assert captured["pgpass"] == "example.com:6543:demo:alice:secret\n"
    pgpass_path = captured["pgpass_path"]
    assert isinstance(pgpass_path, Path)
    assert not pgpass_path.exists()
    assert captured["capture_output"] is True
    assert captured["check"] is False


def test_restore_sql_file_removes_pgpass_when_psql_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dump_path = tmp_path / "dump.sql"
    dump_path.write_bytes(b"select 1;\n")
    pgpass_paths: list[Path] = []

    def fake_run(
        command: list[str],
        *,
        stdin: BinaryIO,
        env: dict[str, str],
        capture_output: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[bytes]:
        _ = (stdin, capture_output, check)
        pgpass_paths.append(Path(env["PGPASSFILE"]))
        return subprocess.CompletedProcess(command, 1, b"", b"syntax error")

    monkeypatch.setattr(postgres_sync_module.subprocess, "run", fake_run)

    with pytest.raises(
        ProjectError, match="psql restore failed: syntax error"
    ):
        postgres_sync_module._restore_sql_file(
            "postgresql://alice:secret@example.com/demo",
            dump_path,
        )

    assert len(pgpass_paths) == 1
    assert not pgpass_paths[0].exists()


def test_restore_sql_file_translates_missing_psql(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dump_path = tmp_path / "dump.sql"
    dump_path.write_bytes(b"select 1;\n")

    def fake_run(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError("psql")

    monkeypatch.setattr(postgres_sync_module.subprocess, "run", fake_run)

    with pytest.raises(ProjectError, match="psql is required"):
        postgres_sync_module._restore_sql_file(
            "postgresql://alice:secret@example.com/demo",
            dump_path,
        )


def test_raise_for_psql_restore_failure_uses_unknown_error() -> None:
    result = subprocess.CompletedProcess(["psql"], 1, b"", b"")

    with pytest.raises(ProjectError, match="unknown error"):
        postgres_sync_module._raise_for_psql_restore_failure(result)


class FakeCursor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    existing_databases: set[str] = Field(default_factory=set)
    operations: list[tuple[str, str]] = Field(default_factory=list)

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, query: object, params: list[str] | None = None) -> None:
        _ = query
        if params is not None:
            self.operations.append(("terminate", params[0]))

    def fetchone(self) -> tuple[int] | None:
        return None


class FakeConnection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cursor_value: FakeCursor

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def cursor(self) -> FakeCursor:
        return self.cursor_value


def _fake_admin_operations(
    cursor: FakeCursor,
    *,
    fail_renames: set[tuple[str, str]] | None = None,
) -> postgres_sync_module.PsycopgPostgresAdminOperations:
    failed = fail_renames or set()

    def fake_rename_database(
        cur: postgres_sync_module.AdminCursor,
        source_database: str,
        target_database: str,
    ) -> None:
        _ = cur
        if (source_database, target_database) in failed:
            raise RuntimeError(
                f"rename {source_database} to {target_database} failed"
            )
        cursor.operations.append(
            ("rename", f"{source_database}->{target_database}")
        )

    def fake_database_exists(
        cur: postgres_sync_module.AdminCursor,
        database_name: str,
    ) -> bool:
        _ = cur
        return database_name in cursor.existing_databases

    def fake_terminate_connections(
        cur: postgres_sync_module.AdminCursor,
        database_name: str,
    ) -> None:
        _ = cur
        cursor.operations.append(("terminate", database_name))

    return postgres_sync_module.PsycopgPostgresAdminOperations(
        connect_admin=lambda admin_url: FakeConnection(cursor_value=cursor),
        database_exists=fake_database_exists,
        rename_database=fake_rename_database,
        terminate_connections=fake_terminate_connections,
    )


def test_replace_database_renames_temporary_when_target_is_absent() -> None:
    cursor = FakeCursor()
    admin = _fake_admin_operations(cursor)

    replaced_existing = admin.replace_database(
        postgres_sync_module.DatabaseSwapPlan(
            admin_url="postgresql://example/postgres",
            temporary_database="demo_sync",
            target_database="demo",
            previous_database="demo_prev",
        )
    )

    assert replaced_existing is False
    assert cursor.operations == [("rename", "demo_sync->demo")]


def test_replace_database_moves_existing_target_before_swap() -> None:
    cursor = FakeCursor(existing_databases={"demo"})
    admin = _fake_admin_operations(cursor)

    replaced_existing = admin.replace_database(
        postgres_sync_module.DatabaseSwapPlan(
            admin_url="postgresql://example/postgres",
            temporary_database="demo_sync",
            target_database="demo",
            previous_database="demo_prev",
        )
    )

    assert replaced_existing is True
    assert cursor.operations == [
        ("terminate", "demo"),
        ("rename", "demo->demo_prev"),
        ("rename", "demo_sync->demo"),
    ]


def test_replace_database_restores_previous_when_swap_fails() -> None:
    cursor = FakeCursor(existing_databases={"demo"})
    admin = _fake_admin_operations(
        cursor,
        fail_renames={("demo_sync", "demo")},
    )

    with pytest.raises(RuntimeError, match="rename demo_sync to demo failed"):
        admin.replace_database(
            postgres_sync_module.DatabaseSwapPlan(
                admin_url="postgresql://example/postgres",
                temporary_database="demo_sync",
                target_database="demo",
                previous_database="demo_prev",
            )
        )

    assert cursor.operations == [
        ("terminate", "demo"),
        ("rename", "demo->demo_prev"),
        ("terminate", "demo_prev"),
        ("rename", "demo_prev->demo"),
    ]


def test_replace_database_translates_failed_rollback() -> None:
    cursor = FakeCursor(existing_databases={"demo"})
    admin = _fake_admin_operations(
        cursor,
        fail_renames={
            ("demo_sync", "demo"),
            ("demo_prev", "demo"),
        },
    )

    with pytest.raises(ProjectError, match="rollback also failed"):
        admin.replace_database(
            postgres_sync_module.DatabaseSwapPlan(
                admin_url="postgresql://example/postgres",
                temporary_database="demo_sync",
                target_database="demo",
                previous_database="demo_prev",
            )
        )

    assert cursor.operations == [
        ("terminate", "demo"),
        ("rename", "demo->demo_prev"),
        ("terminate", "demo_prev"),
    ]
