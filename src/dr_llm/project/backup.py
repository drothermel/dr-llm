from __future__ import annotations

import gzip
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dr_llm.project.models import ProjectInfo

DEFAULT_BACKUP_DIR = Path.home() / ".dr-llm" / "backups"


def _psql(cname: str, db: str, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [
            "docker",
            "exec",
            cname,
            "psql",
            "-U",
            ProjectInfo.db_user,
            db,
            *args,
        ],
        capture_output=True,
        check=True,
    )


def backup_project(name: str, output_dir: Path | None = None) -> Path:
    project_info = ProjectInfo.get_by_name(name)
    if not project_info.running:
        raise RuntimeError(
            f"Project '{name}' is {project_info.status} — start it before backing up"
        )

    backup_dir = (output_dir or DEFAULT_BACKUP_DIR) / name
    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"{name}_{timestamp}.sql.gz"

    result = subprocess.run(
        [
            "docker",
            "exec",
            project_info.container_name,
            "pg_dump",
            "-U",
            ProjectInfo.db_user,
            ProjectInfo.db_name,
        ],
        capture_output=True,
        check=True,
    )

    with gzip.open(backup_file, "wb") as f:
        f.write(result.stdout)

    return backup_file


def restore_project(name: str, backup_file: Path) -> None:
    project_info = ProjectInfo.get_by_name(name)
    if not project_info.running:
        raise RuntimeError(
            f"Project '{name}' is {project_info.status} — start it before restoring"
        )

    if not backup_file.exists():
        raise FileNotFoundError(f"Backup file not found: {backup_file}")

    if backup_file.suffix == ".gz":
        with gzip.open(backup_file, "rb") as f:
            sql_bytes = f.read()
    else:
        sql_bytes = backup_file.read_bytes()

    cname = project_info.container_name
    tmp_db = f"dr_llm_restore_{uuid4().hex[:8]}"

    # Restore into a temp database first so the original is untouched on failure.
    _psql(cname, "postgres", "-c", f"CREATE DATABASE {tmp_db};")
    try:
        subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                cname,
                "psql",
                "-U",
                project_info.db_name,
                tmp_db,
            ],
            input=sql_bytes,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        _psql(cname, "postgres", "-c", f"DROP DATABASE IF EXISTS {tmp_db};")
        raise

    # Swap: drop original, rename temp to original.
    _psql(
        cname,
        "postgres",
        "-c",
        f"DROP DATABASE IF EXISTS {project_info.db_name};",
        "-c",
        f"ALTER DATABASE {tmp_db} RENAME TO {project_info.db_name};",
    )
