from __future__ import annotations

from pathlib import Path

import pytest

import dr_llm.project.postgres_sync as postgres_sync_module
from dr_llm.project.docker_project_metadata import ContainerStatus
from dr_llm.project.errors import ProjectError
from dr_llm.project.models import ProjectSyncValidation
from dr_llm.project.project_info import ProjectInfo


def test_sync_project_to_postgres_validates_and_swaps_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object]] = []

    monkeypatch.setattr(
        postgres_sync_module,
        "get_project",
        lambda name: ProjectInfo(
            name=name, port=5500, status=ContainerStatus.RUNNING
        ),
    )
    monkeypatch.setattr(
        postgres_sync_module,
        "_create_database",
        lambda admin_url, database_name: events.append(
            ("create", database_name)
        ),
    )

    def fake_dump_project_to_file(name: str, dump_path: Path) -> None:
        events.append(("dump", name))
        dump_path.write_text("select 1;\n")

    monkeypatch.setattr(
        postgres_sync_module,
        "_dump_project_to_file",
        fake_dump_project_to_file,
    )
    monkeypatch.setattr(
        postgres_sync_module,
        "_restore_sql_file",
        lambda target_dsn, dump_path: events.append(("restore", target_dsn)),
    )
    monkeypatch.setattr(
        postgres_sync_module,
        "validate_project_database_copy",
        lambda *, source_dsn, target_dsn: (
            events.append(("validate", (source_dsn, target_dsn)))
            or ProjectSyncValidation(
                source_table_count=1,
                target_table_count=1,
                checked_table_count=1,
            )
        ),
    )
    monkeypatch.setattr(
        postgres_sync_module,
        "_replace_database",
        lambda **kwargs: (
            events.append(("replace", kwargs["target_database"])) or True
        ),
    )
    monkeypatch.setattr(
        postgres_sync_module,
        "_drop_database",
        lambda admin_url, database_name: events.append(
            ("drop", database_name)
        ),
    )

    result = postgres_sync_module.sync_project_to_postgres(
        "demo",
        "postgresql://example/postgres?sslmode=require",
        drop_previous=True,
    )

    assert result.success is True
    assert result.target_database == "demo"
    assert result.previous_database is not None
    assert result.previous_database_dropped is True
    assert [event[0] for event in events] == [
        "create",
        "dump",
        "restore",
        "validate",
        "replace",
        "drop",
    ]
    assert ("dump", "demo") in events


def test_sync_project_to_postgres_drops_temp_database_when_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dropped: list[str] = []

    monkeypatch.setattr(
        postgres_sync_module,
        "get_project",
        lambda name: ProjectInfo(
            name=name, port=5500, status=ContainerStatus.RUNNING
        ),
    )
    monkeypatch.setattr(
        postgres_sync_module, "_create_database", lambda *args: None
    )
    monkeypatch.setattr(
        postgres_sync_module,
        "_dump_project_to_file",
        lambda name, dump_path: dump_path.write_text("select 1;\n"),
    )
    monkeypatch.setattr(
        postgres_sync_module, "_restore_sql_file", lambda *args: None
    )
    monkeypatch.setattr(
        postgres_sync_module,
        "validate_project_database_copy",
        lambda *, source_dsn, target_dsn: ProjectSyncValidation(
            source_table_count=1,
            target_table_count=1,
            checked_table_count=1,
            mismatches=["alpha row count mismatch: source=1 target=0"],
        ),
    )
    monkeypatch.setattr(
        postgres_sync_module,
        "_drop_database",
        lambda admin_url, database_name: dropped.append(database_name),
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


def test_url_for_database_preserves_connection_options() -> None:
    assert (
        postgres_sync_module._url_for_database(
            "postgresql://user:pass@example/postgres?sslmode=require",
            "target_db",
        )
        == "postgresql://user:pass@example/target_db?sslmode=require"
    )


def test_url_for_database_requires_scheme_and_host() -> None:
    with pytest.raises(ProjectError, match="scheme and host"):
        postgres_sync_module._url_for_database("postgres", "target_db")
