from __future__ import annotations

import socket
import subprocess

from dr_llm.project.project_info import LABEL_PREFIX, parse_docker_labels

BASE_PORT = 5500


def _port_is_free(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", port))
            return True
    except OSError:
        return False


def _claimed_ports() -> set[int]:
    result = subprocess.run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"label={LABEL_PREFIX}.name",
            "--format",
            "{{json .Labels}}",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to list Docker containers: {result.stderr.strip()}")
    ports: set[int] = set()
    for line in result.stdout.strip().splitlines():
        labels = parse_docker_labels(line)
        port_str = labels.get(f"{LABEL_PREFIX}.port")
        if port_str is not None:
            try:
                ports.add(int(port_str))
            except ValueError:
                continue
    return ports


def find_available_port(base: int = BASE_PORT, max_attempts: int = 100) -> int:
    claimed = _claimed_ports()
    for offset in range(max_attempts):
        port = base + offset
        if port not in claimed and _port_is_free(port):
            return port
    raise RuntimeError(
        f"No available port found in range {base}–{base + max_attempts - 1}"
    )
