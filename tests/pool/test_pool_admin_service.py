from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from dr_llm.pool import admin_service
from dr_llm.pool.db.schema import KeyColumn, PoolSchema
from dr_llm.pool.errors import PoolError
from dr_llm.pool.models import (
    CreatePoolRequest,
    DeletePoolRequest,
    DeletePoolsByTokenRequest,
    PoolCreationBlockReason,
    PoolCreationReadiness,
    PoolDeletionBlockReason,
    PoolDeletionResult,
    PoolDeletionStatus,
    PoolInspection,
    PoolInspectionRequest,
    PoolInspectionStatus,
)
from dr_llm.pool.pending.pending_status import PendingStatusCounts
from dr_llm.project.docker_project_metadata import ContainerStatus
from dr_llm.project.errors import ProjectNotFoundError
from dr_llm.project.project_info import ProjectInfo


def test_create_pool_request_from_csv_normalizes_axes() -> None:
    request = CreatePoolRequest.from_csv(
        project_name=" demo ",
        pool_name=" sample_pool ",
        axes_csv=" provider, model ,, region ",
    )

    assert request.project_name == "demo"
    assert request.pool_name == "sample_pool"
    assert request.key_axes == ["provider", "model", "region"]


def test_assess_pool_creation_reports_request_violations_before_project_lookup() -> (
    None
):
    readiness = admin_service.assess_pool_creation(
        CreatePoolRequest(project_name="demo", pool_name="Bad Name", key_axes=[]),
    )

    assert readiness.allowed is False
    assert {violation.reason for violation in readiness.violations} == {
        PoolCreationBlockReason.invalid_pool_name,
        PoolCreationBlockReason.missing_key_axes,
        PoolCreationBlockReason.project_not_found,
    }
    assert readiness.blocked_message is not None
    assert "pool_name must be lowercase alphanumeric" in readiness.blocked_message
    assert "At least one key axis is required" in readiness.blocked_message
    assert "Project 'demo' not found" in readiness.blocked_message


def test_pool_creation_readiness_blocked_message_is_none_when_allowed() -> None:
    readiness = PoolCreationReadiness(
        request=CreatePoolRequest(
            project_name="demo",
            pool_name="sample_pool",
            key_axes=["provider"],
        )
    )

    assert readiness.allowed is True
    assert readiness.blocked_message is None


def test_assess_pool_creation_reports_project_not_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        admin_service,
        "maybe_get_project",
        lambda name: ProjectInfo(name=name, status=ContainerStatus.STOPPED),
    )

    readiness = admin_service.assess_pool_creation(
        CreatePoolRequest(
            project_name="demo", pool_name="sample_pool", key_axes=["provider"]
        )
    )

    assert readiness.allowed is False
    assert [violation.reason for violation in readiness.violations] == [
        PoolCreationBlockReason.project_not_running
    ]


def test_assess_pool_creation_reports_existing_pool_limits_progress_and_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = ProjectInfo(name="demo", port=5500, status=ContainerStatus.RUNNING)
    monkeypatch.setattr(admin_service, "maybe_get_project", lambda name: project)
    monkeypatch.setattr(admin_service, "discover_pools", lambda dsn: ["alpha", "beta"])
    recent = datetime.now(UTC)
    monkeypatch.setattr(
        admin_service,
        "_inspect_pool_for_project",
        lambda project, pool_name: PoolInspection(
            project_name=project.name,
            name=pool_name,
            pool_schema=PoolSchema(
                name=pool_name,
                key_columns=[KeyColumn(name="provider")],
            ),
            created_at=recent,
            sample_count=1,
            pending_counts=PendingStatusCounts(
                pending=1 if pool_name == "alpha" else 0
            ),
            status=(
                PoolInspectionStatus.in_progress
                if pool_name == "alpha"
                else PoolInspectionStatus.complete
            ),
        ),
    )

    readiness = admin_service.assess_pool_creation(
        CreatePoolRequest(
            project_name="demo", pool_name="alpha", key_axes=["provider"]
        ),
        max_pools_per_project=2,
        cooldown_seconds=60,
    )

    assert readiness.allowed is False
    assert {violation.reason for violation in readiness.violations} == {
        PoolCreationBlockReason.pool_already_exists,
        PoolCreationBlockReason.max_pools_reached,
        PoolCreationBlockReason.pool_in_progress,
        PoolCreationBlockReason.cooldown_active,
    }


def test_inspect_pool_derives_status_from_reader_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = ProjectInfo(name="demo", port=5500, status=ContainerStatus.RUNNING)
    schema = PoolSchema(name="sample_pool", key_columns=[KeyColumn(name="provider")])
    closed: list[bool] = []

    class FakeRuntime:
        def __init__(self, config: object) -> None:
            _ = config

        def connect(self) -> object:
            class _Conn:
                def __enter__(self) -> _Conn:
                    return self

                def __exit__(self, *exc_info: object) -> None:
                    return None

                def execute(self, stmt: object) -> SimpleNamespace:
                    _ = stmt
                    return SimpleNamespace(
                        scalar_one_or_none=lambda: datetime(2024, 1, 2, tzinfo=UTC)
                    )

            return _Conn()

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(admin_service, "maybe_get_project", lambda name: project)
    monkeypatch.setattr(admin_service, "DbRuntime", FakeRuntime)
    monkeypatch.setattr(
        admin_service, "load_schema_from_db", lambda runtime, pool_name: schema
    )
    monkeypatch.setattr(
        admin_service.PoolReader,
        "from_runtime",
        lambda runtime, schema: SimpleNamespace(
            progress=lambda: SimpleNamespace(
                samples_total=2,
                pending_counts=PendingStatusCounts(pending=1, leased=0, failed=0),
            )
        ),
    )

    inspection = admin_service.inspect_pool(
        PoolInspectionRequest(project_name="demo", pool_name="sample_pool")
    )

    assert inspection.name == "sample_pool"
    assert inspection.status == PoolInspectionStatus.in_progress
    assert inspection.created_at == datetime(2024, 1, 2, tzinfo=UTC)
    assert closed == [True]


def test_create_pool_ensures_schema_and_returns_fresh_inspection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = ProjectInfo(name="demo", port=5500, status=ContainerStatus.RUNNING)
    request = CreatePoolRequest(
        project_name="demo",
        pool_name="sample_pool",
        key_axes=["provider"],
    )
    readiness = PoolCreationReadiness(request=request, project=project)
    ensured: list[PoolSchema] = []
    closed: list[bool] = []
    inspection = PoolInspection(
        project_name="demo",
        name="sample_pool",
        pool_schema=PoolSchema(
            name="sample_pool",
            key_columns=[KeyColumn(name="provider")],
        ),
        created_at=None,
        sample_count=0,
        pending_counts=PendingStatusCounts(),
        status=PoolInspectionStatus.empty,
    )

    class FakeRuntime:
        def __init__(self, config: object) -> None:
            _ = config

        def close(self) -> None:
            closed.append(True)

    class FakeStore:
        def __init__(self, schema: PoolSchema, runtime: FakeRuntime) -> None:
            _ = runtime
            self.schema = schema

        def ensure_schema(self) -> None:
            ensured.append(self.schema)

    monkeypatch.setattr(
        admin_service, "assess_pool_creation", lambda request: readiness
    )
    monkeypatch.setattr(admin_service, "DbRuntime", FakeRuntime)
    monkeypatch.setattr(admin_service, "PoolStore", FakeStore)
    monkeypatch.setattr(admin_service, "inspect_pool", lambda request: inspection)

    created = admin_service.create_pool(request)

    assert created == inspection
    assert ensured == [inspection.pool_schema]
    assert closed == [True]


def test_create_pool_raises_project_not_found_for_missing_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = CreatePoolRequest(
        project_name="missing",
        pool_name="sample_pool",
        key_axes=["provider"],
    )
    readiness = PoolCreationReadiness(
        request=request,
        violations=[
            admin_service.PoolCreationViolation(
                reason=PoolCreationBlockReason.project_not_found,
                message="Project 'missing' not found",
                project_name="missing",
                pool_name="sample_pool",
            )
        ],
    )
    monkeypatch.setattr(
        admin_service, "assess_pool_creation", lambda request: readiness
    )

    with pytest.raises(ProjectNotFoundError, match="Project 'missing' not found"):
        admin_service.create_pool(request)


def test_create_pool_raises_pool_error_for_blocked_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = CreatePoolRequest(
        project_name="demo",
        pool_name="sample_pool",
        key_axes=["provider"],
    )
    readiness = PoolCreationReadiness(
        request=request,
        project=ProjectInfo(name="demo", port=5500, status=ContainerStatus.RUNNING),
        violations=[
            admin_service.PoolCreationViolation(
                reason=PoolCreationBlockReason.pool_already_exists,
                message="Pool 'sample_pool' already exists in project 'demo'.",
                project_name="demo",
                pool_name="sample_pool",
            )
        ],
    )
    monkeypatch.setattr(
        admin_service, "assess_pool_creation", lambda request: readiness
    )

    with pytest.raises(PoolError, match="already exists"):
        admin_service.create_pool(request)


def test_assess_pool_deletion_reports_project_not_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        admin_service,
        "maybe_get_project",
        lambda name: ProjectInfo(name=name, status=ContainerStatus.STOPPED),
    )

    readiness = admin_service.assess_pool_deletion(
        DeletePoolRequest(project_name="demo", pool_name="sample_pool")
    )

    assert readiness.allowed is False
    assert [violation.reason for violation in readiness.violations] == [
        PoolDeletionBlockReason.project_not_running
    ]


def test_assess_pool_deletion_reports_missing_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = ProjectInfo(name="demo", port=5500, status=ContainerStatus.RUNNING)
    monkeypatch.setattr(admin_service, "maybe_get_project", lambda name: project)
    monkeypatch.setattr(admin_service, "_existing_pool_table_names", lambda *args: [])
    monkeypatch.setattr(
        admin_service, "_count_in_progress_pending_rows", lambda *args: 0
    )

    class FakeRuntime:
        def __init__(self, config: object) -> None:
            _ = config

        def close(self) -> None:
            return None

    monkeypatch.setattr(admin_service, "DbRuntime", FakeRuntime)

    readiness = admin_service.assess_pool_deletion(
        DeletePoolRequest(project_name="demo", pool_name="sample_pool")
    )

    assert readiness.allowed is False
    assert [violation.reason for violation in readiness.violations] == [
        PoolDeletionBlockReason.pool_not_found
    ]


def test_assess_pool_deletion_allows_pending_rows_but_reports_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = ProjectInfo(name="demo", port=5500, status=ContainerStatus.RUNNING)
    monkeypatch.setattr(admin_service, "maybe_get_project", lambda name: project)
    monkeypatch.setattr(
        admin_service,
        "_existing_pool_table_names",
        lambda *args: ["pool_sample_pool_samples", "pool_sample_pool_pending"],
    )
    monkeypatch.setattr(
        admin_service,
        "_count_in_progress_pending_rows",
        lambda *args: 3,
    )

    class FakeRuntime:
        def __init__(self, config: object) -> None:
            _ = config

        def close(self) -> None:
            return None

    monkeypatch.setattr(admin_service, "DbRuntime", FakeRuntime)

    readiness = admin_service.assess_pool_deletion(
        DeletePoolRequest(project_name="demo", pool_name="sample_pool")
    )

    assert readiness.allowed is True
    assert readiness.in_progress_pending_count == 3
    assert readiness.violations == []


def test_pool_name_has_token_match_uses_exact_underscore_delimited_words() -> None:
    assert admin_service.pool_name_has_token_match("smoke_eval")
    assert admin_service.pool_name_has_token_match("demo_tst_case")
    assert admin_service.pool_name_has_token_match("alpha_demo_run")
    assert admin_service.pool_name_has_token_match("full_test_run")
    assert not admin_service.pool_name_has_token_match("contest_pool")
    assert not admin_service.pool_name_has_token_match("smoketest_pool")
    assert not admin_service.pool_name_has_token_match("demographic_pool")
    assert not admin_service.pool_name_has_token_match("attest_case")


def test_delete_pools_by_token_matches_only_exact_words_in_discovery_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = ProjectInfo(name="demo", port=5500, status=ContainerStatus.RUNNING)
    deleted: list[str] = []

    def fake_delete_pool(request: DeletePoolRequest) -> PoolDeletionResult:
        deleted.append(request.pool_name)
        return PoolDeletionResult(
            request=request,
            status=PoolDeletionStatus.deleted,
        )

    monkeypatch.setattr(admin_service, "maybe_get_project", lambda name: project)
    monkeypatch.setattr(
        admin_service,
        "discover_pools",
        lambda dsn: [
            "alpha_test_run",
            "contest_pool",
            "smoke_eval",
            "demo_tst_case",
            "alpha_demo_run",
            "smoketest_pool",
        ],
    )
    monkeypatch.setattr(admin_service, "delete_pool", fake_delete_pool)

    result = admin_service.delete_pools_by_token(
        DeletePoolsByTokenRequest(
            project_name="demo",
            match_tokens=["test", "tst", "smoke", "demo"],
        )
    )

    assert result.success is True
    assert result.matched_pool_names == [
        "alpha_test_run",
        "smoke_eval",
        "demo_tst_case",
        "alpha_demo_run",
    ]
    assert deleted == result.matched_pool_names


def test_delete_pools_by_token_dry_run_reports_matches_without_deleting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = ProjectInfo(name="demo", port=5500, status=ContainerStatus.RUNNING)
    deleted: list[str] = []

    def fake_delete_pool(request: DeletePoolRequest) -> PoolDeletionResult:
        deleted.append(request.pool_name)
        return PoolDeletionResult(
            request=request,
            status=PoolDeletionStatus.deleted,
        )

    monkeypatch.setattr(admin_service, "maybe_get_project", lambda name: project)
    monkeypatch.setattr(
        admin_service,
        "discover_pools",
        lambda dsn: ["alpha_test_run", "contest_pool", "smoke_eval", "alpha_demo_run"],
    )
    monkeypatch.setattr(admin_service, "delete_pool", fake_delete_pool)

    result = admin_service.delete_pools_by_token(
        DeletePoolsByTokenRequest(
            project_name="demo",
            match_tokens=["test", "tst", "smoke", "demo"],
            dry_run=True,
        )
    )

    assert result.success is True
    assert result.dry_run is True
    assert result.matched_pool_names == [
        "alpha_test_run",
        "smoke_eval",
        "alpha_demo_run",
    ]
    assert result.pool_results == []
    assert deleted == []
