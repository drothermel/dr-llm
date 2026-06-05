from __future__ import annotations

import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import threading


def test_run_tests_local_cleans_up_after_pytest_failure(
    tmp_path: Path,
) -> None:
    call_log = tmp_path / "calls.log"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    listener, port = _listening_socket()
    thread = threading.Thread(
        target=_accept_one_connection, args=(listener,), daemon=True
    )
    thread.start()
    _write_fake_docker(fake_bin / "docker", call_log)
    _write_fake_uv(fake_bin / "uv", call_log)
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "run-tests-local.sh"
    )
    env = {
        **os.environ,
        "CALL_LOG": str(call_log),
        "FAKE_NATS_PORT": str(port),
        "NATS_READY_TIMEOUT_SECONDS": "1",
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "PYTHON_FOR_UV_STUB": sys.executable,
    }

    result = subprocess.run(
        ["bash", str(script_path)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    thread.join(timeout=1)

    assert result.returncode == 7
    calls = call_log.read_text(encoding="utf-8").splitlines()
    pytest_index = calls.index("uv run pytest tests/ -v -m integration -n 0")
    assert _has_call_after(
        calls, "docker rm -f dr_llm_nats_test_runner", pytest_index
    )
    assert _has_call_after(
        calls,
        "uv run dr-llm project destroy dr_llm_test_runner "
        "--yes-really-delete-everything",
        pytest_index,
    )


def _listening_socket() -> tuple[socket.socket, int]:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    return listener, listener.getsockname()[1]


def _has_call_after(calls: list[str], expected: str, index: int) -> bool:
    return any(call == expected for call in calls[index + 1 :])


def _accept_one_connection(listener: socket.socket) -> None:
    try:
        connection, _ = listener.accept()
        connection.close()
    finally:
        listener.close()


def _write_fake_docker(path: Path, call_log: Path) -> None:
    quoted_call_log = shlex.quote(str(call_log))
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'echo "docker $*" >> {quoted_call_log}\n'
        'case "$1" in\n'
        '  port) echo "127.0.0.1:${FAKE_NATS_PORT}" ;;\n'
        "esac\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_fake_uv(path: Path, call_log: Path) -> None:
    quoted_call_log = shlex.quote(str(call_log))
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'echo "uv $*" >> {quoted_call_log}\n'
        'if [ "$1" = "run" ] && [ "$2" = "python" ]; then\n'
        "  shift 2\n"
        '  "${PYTHON_FOR_UV_STUB}" "$@"\n'
        "  exit $?\n"
        "fi\n"
        'if [ "$1" = "run" ] && [ "$2" = "pytest" ]; then\n'
        "  exit 7\n"
        "fi\n"
        'if [ "$1" = "run" ] && [ "$2" = "dr-llm" ] '
        '&& [ "$3" = "project" ] && [ "$4" = "create" ]; then\n'
        '  echo \'{"dsn":"postgresql://example/test"}\'\n'
        "fi\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
