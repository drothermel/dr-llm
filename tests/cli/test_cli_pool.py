from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

import dr_llm.cli.pool as pool_cli
from dr_llm.cli import app
from dr_llm.pool.models import DeletePoolRequest, PoolDeletionResult, PoolDeletionStatus

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
