from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from dr_llm.env import find_dotenv, load_dotenv


def test_load_dotenv_sets_values_without_overriding_existing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tracked_keys = ["PLAIN", "EXPORTED", "EXISTING"]
    original_values = {key: os.environ.get(key) for key in tracked_keys}
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "# ignored",
                "PLAIN=value",
                "export EXPORTED='quoted value'",
                'EXISTING="from-file"',
            ]
        )
    )
    monkeypatch.setenv("EXISTING", "from-env")

    try:
        load_dotenv(env_path)

        assert os.environ["PLAIN"] == "value"
        assert os.environ["EXPORTED"] == "quoted value"
        assert os.environ["EXISTING"] == "from-env"
    finally:
        for key, value in original_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_find_dotenv_walks_up_from_nested_directory(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("DR_LLM_DATABASE_URL=postgresql://example/db\n")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)

    assert find_dotenv(nested) == env_path


def test_cli_import_loads_dotenv_before_command_parsing(
    tmp_path: Path,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "DR_LLM_POSTGRES_SYNC_ADMIN_URL=postgresql://example/neondb\n"
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if key != "DR_LLM_POSTGRES_SYNC_ADMIN_URL"
    }

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; import dr_llm.cli.app; "
                "print(os.environ.get('DR_LLM_POSTGRES_SYNC_ADMIN_URL', ''))"
            ),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "postgresql://example/neondb"
