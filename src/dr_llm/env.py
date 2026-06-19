from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path | None = None, *, override: bool = False) -> None:
    """Load variables from a .env file into os.environ.

    Uses path when provided, otherwise the nearest .env found by find_dotenv.
    Existing environment variables are preserved unless override is true.
    """
    env_path = path or find_dotenv()
    if env_path is None or not env_path.exists():
        return

    for raw_line in env_path.read_text(errors="ignore").splitlines():
        parsed = _parse_env_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        if not override and key in os.environ:
            continue
        os.environ[key] = value


def find_dotenv(start: Path | None = None) -> Path | None:
    """Return the nearest .env path by walking upward from start or cwd."""
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for directory in (current, *current.parents):
        env_path = directory / ".env"
        if env_path.exists():
            return env_path
    return None


def _parse_env_line(raw_line: str) -> tuple[str, str] | None:
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None

    key, value = line.split("=", 1)
    key = key.strip()
    if key.startswith("export "):
        key = key.removeprefix("export ").strip()
    if not key:
        return None

    return key, _clean_env_value(value)


def _clean_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
