from __future__ import annotations

from pathlib import Path


def test_run_tests_local_installs_cleanup_trap_before_resource_creation() -> (
    None
):
    script = (
        Path(__file__).resolve().parents[2] / "scripts" / "run-tests-local.sh"
    ).read_text(encoding="utf-8")

    trap_index = script.index("trap cleanup EXIT")
    nats_create_index = script.index("Creating temporary NATS container")
    project_create_index = script.index("Creating temporary project")

    assert trap_index < nats_create_index
    assert trap_index < project_create_index
