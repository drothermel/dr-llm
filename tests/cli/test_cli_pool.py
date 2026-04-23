from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

import dr_llm.cli.pool as pool_cli
from dr_llm.cli import app
from dr_llm.pool.models import (
    DeletePoolRequest,
    DeletePoolsByTokenRequest,
    DeletePoolsByTokenResult,
    DeletePoolsByTokenStatus,
    PoolDeletionResult,
    PoolDeletionStatus,
)

runner = CliRunner()


def test_pool_destroy_invokes_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deleted: list[tuple[str, str]] = []

    def fake_delete_pool(request: DeletePoolRequest) -> PoolDeletionResult:
        deleted.append((request.project_name, request.pool_name))
        return PoolDeletionResult(
            request=request,
            status=PoolDeletionStatus.deleted,
            deleted_table_names=["pool_sample_pool_samples"],
        )

    monkeypatch.setattr(pool_cli, "delete_pool", fake_delete_pool)

    result = runner.invoke(
        app,
        [
            "pool",
            "destroy",
            "demo",
            "sample_pool",
            "--yes-really-delete-everything",
        ],
    )

    assert result.exit_code == 0
    assert deleted == [("demo", "sample_pool")]
    payload = json.loads(result.stdout)
    assert payload["status"] == "deleted"
    assert payload["request"] == {
        "project_name": "demo",
        "pool_name": "sample_pool",
    }


def test_pool_destroy_requires_confirmation_without_flag() -> None:
    result = runner.invoke(app, ["pool", "destroy", "demo", "sample_pool"], input="n\n")

    assert result.exit_code == 1
    assert "Continue?" in result.output


def test_pool_destroy_testish_invokes_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deleted: list[tuple[str, list[str]]] = []

    def fake_delete_pools_by_token(
        request: DeletePoolsByTokenRequest,
    ) -> DeletePoolsByTokenResult:
        deleted.append((request.project_name, request.match_tokens))
        return DeletePoolsByTokenResult(
            request=request,
            status=DeletePoolsByTokenStatus.completed,
            discovered_pool_names=["alpha_test_run", "contest_pool"],
            matched_pool_names=["alpha_test_run"],
            pool_results=[
                PoolDeletionResult(
                    request=DeletePoolRequest(
                        project_name=request.project_name,
                        pool_name="alpha_test_run",
                    ),
                    status=PoolDeletionStatus.deleted,
                )
            ],
        )

    monkeypatch.setattr(pool_cli, "delete_pools_by_token", fake_delete_pools_by_token)

    result = runner.invoke(
        app,
        ["pool", "destroy-testish", "demo", "--yes-really-delete-everything"],
    )

    assert result.exit_code == 0
    assert deleted == [("demo", ["test", "tst", "smoke", "demo"])]
    payload = json.loads(result.stdout)
    assert payload["status"] == "completed"
    assert payload["matched_pool_names"] == ["alpha_test_run"]


def test_pool_destroy_testish_dry_run_invokes_service_without_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deleted: list[tuple[str, list[str], bool]] = []

    def fake_delete_pools_by_token(
        request: DeletePoolsByTokenRequest,
    ) -> DeletePoolsByTokenResult:
        deleted.append((request.project_name, request.match_tokens, request.dry_run))
        return DeletePoolsByTokenResult(
            request=request,
            status=DeletePoolsByTokenStatus.completed,
            discovered_pool_names=["alpha_test_run", "contest_pool"],
            matched_pool_names=["alpha_test_run"],
            dry_run=True,
            message="Dry run: would delete 1 matching pools.",
        )

    monkeypatch.setattr(pool_cli, "delete_pools_by_token", fake_delete_pools_by_token)

    result = runner.invoke(
        app,
        ["pool", "destroy-testish", "demo", "--dry-run"],
    )

    assert result.exit_code == 0
    assert deleted == [("demo", ["test", "tst", "smoke", "demo"], True)]
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert payload["matched_pool_names"] == ["alpha_test_run"]
