from __future__ import annotations

from pathlib import Path

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


class SyncCallRecorder(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operations: list[str] = Field(default_factory=list)
    created_databases: list[str] = Field(default_factory=list)
    dumped_projects: list[str] = Field(default_factory=list)
    restored_dsns: list[str] = Field(default_factory=list)
    validations: list[ValidationCall] = Field(default_factory=list)
    replaced_databases: list[str] = Field(default_factory=list)
    dropped_databases: list[str] = Field(default_factory=list)

    @property
    def operation_names(self) -> list[str]:
        return self.operations

    def create_database(self, admin_url: str, database_name: str) -> None:
        _ = admin_url
        self.operations.append("create")
        self.created_databases.append(database_name)

    def dump_project_to_file(self, name: str, dump_path: Path) -> None:
        self.operations.append("dump")
        self.dumped_projects.append(name)
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
        return ProjectSyncValidation(
            source_table_count=1,
            target_table_count=1,
            checked_table_count=1,
        )

    def replace_database(self, **kwargs: str) -> bool:
        self.operations.append("replace")
        self.replaced_databases.append(kwargs["target_database"])
        return True

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


def _install_successful_sync_fakes(
    monkeypatch: pytest.MonkeyPatch,
) -> SyncCallRecorder:
    calls = SyncCallRecorder()

    monkeypatch.setattr(postgres_sync_module, "get_project", _running_project)
    monkeypatch.setattr(
        postgres_sync_module,
        "_create_database",
        calls.create_database,
    )
    monkeypatch.setattr(
        postgres_sync_module,
        "_dump_project_to_file",
        calls.dump_project_to_file,
    )
    monkeypatch.setattr(
        postgres_sync_module,
        "_restore_sql_file",
        calls.restore_sql_file,
    )
    monkeypatch.setattr(
        postgres_sync_module,
        "validate_project_database_copy",
        calls.validate_project_database_copy,
    )
    monkeypatch.setattr(
        postgres_sync_module,
        "_replace_database",
        calls.replace_database,
    )
    monkeypatch.setattr(
        postgres_sync_module,
        "_drop_database",
        calls.drop_database,
    )
    return calls


def test_build_sync_plan_derives_temporary_database_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(postgres_sync_module, "get_project", _running_project)

    plan = postgres_sync_module._build_sync_plan(
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


def test_sync_project_to_postgres_validates_and_swaps_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_successful_sync_fakes(monkeypatch)

    result = postgres_sync_module.sync_project_to_postgres(
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


def test_sync_project_to_postgres_restores_and_validates_temporary_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_successful_sync_fakes(monkeypatch)

    result = postgres_sync_module.sync_project_to_postgres(
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


def test_sync_project_to_postgres_drops_temp_database_when_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dropped: list[str] = []

    monkeypatch.setattr(postgres_sync_module, "get_project", _running_project)
    monkeypatch.setattr(
        postgres_sync_module, "_create_database", lambda *_args: None
    )
    monkeypatch.setattr(
        postgres_sync_module,
        "_dump_project_to_file",
        lambda _name, dump_path: dump_path.write_text("select 1;\n"),
    )
    monkeypatch.setattr(
        postgres_sync_module, "_restore_sql_file", lambda *_args: None
    )
    monkeypatch.setattr(
        postgres_sync_module,
        "validate_project_database_copy",
        lambda **_kwargs: ProjectSyncValidation(
            source_table_count=1,
            target_table_count=1,
            checked_table_count=1,
            mismatches=["alpha row count mismatch: source=1 target=0"],
        ),
    )
    monkeypatch.setattr(
        postgres_sync_module,
        "_drop_database",
        lambda _admin_url, database_name: dropped.append(database_name),
    )

    with pytest.raises(ProjectError, match="validation failed"):
        postgres_sync_module.sync_project_to_postgres(
            "demo", "postgresql://example/postgres"
        )

    assert len(dropped) == 1
    assert "_sync_" in dropped[0]


def test_sync_project_to_postgres_requires_project_dsn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        postgres_sync_module,
        "get_project",
        lambda name: ProjectInfo(name=name, status=ContainerStatus.STOPPED),
    )

    with pytest.raises(ProjectError, match="has no DSN"):
        postgres_sync_module.sync_project_to_postgres(
            "demo", "postgresql://example/postgres"
        )


def test_sync_project_to_postgres_rejects_invalid_admin_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations: list[str] = []

    monkeypatch.setattr(postgres_sync_module, "get_project", _running_project)
    monkeypatch.setattr(
        postgres_sync_module,
        "_create_database",
        lambda *_args: operations.append("create"),
    )
    monkeypatch.setattr(
        postgres_sync_module,
        "_dump_project_to_file",
        lambda _name, _dump_path: operations.append("dump"),
    )
    monkeypatch.setattr(
        postgres_sync_module,
        "_drop_database",
        lambda _admin_url, _database_name: operations.append("drop"),
    )

    with pytest.raises(ProjectError, match="scheme and host"):
        postgres_sync_module.sync_project_to_postgres("demo", "postgres")

    assert operations == []


def test_pgpass_line_escapes_colons_and_backslashes() -> None:
    line = postgres_sync_module._pgpass_line(
        "db:host",
        "5432",
        "db\\name",
        "user:name",
        r"pa:ss\word",
    )

    assert line == r"db\:host:5432:db\\name:user\:name:pa\:ss\\word" + "\n"
