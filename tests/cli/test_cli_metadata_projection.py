from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import dr_llm.metadata_projection.cli as metadata_cli
from dr_llm.cli import app

runner = CliRunner()


def test_metadata_projection_run_dispatches_options(monkeypatch) -> None:
    calls: list[tuple[int | None, int | None, bool]] = []

    async def fake_run_projector(
        *,
        config,
        max_messages: int | None,
        batch_size: int | None,
        from_start: bool,
    ) -> int:
        del config
        calls.append((max_messages, batch_size, from_start))
        return 3

    monkeypatch.setattr(metadata_cli, "_run_projector", fake_run_projector)

    result = runner.invoke(
        app,
        [
            "metadata-projection",
            "run",
            "--max-messages",
            "3",
            "--batch-size",
            "2",
            "--from-start",
        ],
    )

    assert result.exit_code == 0
    assert calls == [(3, 2, True)]
    assert '"processed": 3' in result.output


def test_metadata_projection_attach_artifacts_uses_index_path(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[Path] = []

    def fake_attach(config) -> int:
        calls.append(config.artifact_index_path)
        return 2

    monkeypatch.setattr(
        metadata_cli, "attach_finalized_artifacts", fake_attach
    )

    result = runner.invoke(
        app,
        [
            "metadata-projection",
            "attach-artifacts",
            "--artifact-index-path",
            str(tmp_path / "index.sqlite"),
        ],
    )

    assert result.exit_code == 0
    assert calls == [tmp_path / "index.sqlite"]
    assert '"attached": 2' in result.output
