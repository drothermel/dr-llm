from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_sync_demo() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "demo-project-sync-postgres.py"
    )
    spec = importlib.util.spec_from_file_location(
        "demo_project_sync_postgres", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_database_url_rewrites_database_and_preserves_options() -> None:
    sync_demo = _load_sync_demo()

    assert (
        sync_demo._database_url(
            "postgresql://user:pass@example/dr_llm?sslmode=require",
            "target_db",
        )
        == "postgresql://user:pass@example/target_db?sslmode=require"
    )


def test_demo_pool_schema_uses_stable_public_name() -> None:
    sync_demo = _load_sync_demo()

    assert sync_demo.POOL_SCHEMA.name == "sync_demo_pool"
