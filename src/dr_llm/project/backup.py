from __future__ import annotations

import gzip
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from dr_llm.project.docker import get_project
from dr_llm.project.models import DB_NAME, DB_USER, container_name

DEFAULT_BACKUP_DIR = Path.home() / ".dr-llm" / "backups"


def backup_project(name: str, output_dir: Path | None = None) -> Path:
    info = get_project(name)
    if info is None:
        raise RuntimeError(f"Project '{name}' not found")
    if info.status != "running":
        raise RuntimeError(f"Project '{name}' is {info.status} — start it before backing up")

    backup_dir = (output_dir or DEFAULT_BACKUP_DIR) / name
    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"{name}_{timestamp}.sql.gz"

    result = subprocess.run(
        ["docker", "exec", container_name(name), "pg_dump", "-U", DB_USER, DB_NAME],
        capture_output=True,
        check=True,
    )

    with gzip.open(backup_file, "wb") as f:
        f.write(result.stdout)

    return backup_file


def restore_project(name: str, backup_file: Path) -> None:
    info = get_project(name)
    if info is None:
        raise RuntimeError(f"Project '{name}' not found")
    if info.status != "running":
        raise RuntimeError(f"Project '{name}' is {info.status} — start it before restoring")

    if not backup_file.exists():
        raise FileNotFoundError(f"Backup file not found: {backup_file}")

    if backup_file.suffix == ".gz":
        with gzip.open(backup_file, "rb") as f:
            sql_bytes = f.read()
    else:
        sql_bytes = backup_file.read_bytes()

    cname = container_name(name)

    # Drop and recreate the database so the restore starts clean.
    # Connect to the 'postgres' maintenance db to avoid "in use" errors.
    subprocess.run(
        [
            "docker", "exec", cname, "psql", "-U", DB_USER, "postgres",
            "-c", f"DROP DATABASE IF EXISTS {DB_NAME};",
            "-c", f"CREATE DATABASE {DB_NAME};",
        ],
        capture_output=True,
        check=True,
    )

    subprocess.run(
        ["docker", "exec", "-i", cname, "psql", "-U", DB_USER, DB_NAME],
        input=sql_bytes,
        capture_output=True,
        check=True,
    )
