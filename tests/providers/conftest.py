from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from typing import Any

import httpx


def make_http_client(response_json: dict[str, Any]) -> tuple[dict[str, Any], httpx.Client]:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        captured["headers"] = dict(request.headers)
        captured["url"] = str(request.url)
        return httpx.Response(status_code=200, json=response_json)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return captured, client


def make_subprocess_mock(
    stdout: str,
    returncode: int = 0,
    stderr: str = "",
) -> tuple[dict[str, Any], Callable[..., subprocess.CompletedProcess[str]]]:
    captured: dict[str, Any] = {}

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["command"] = args[0]
        captured["input"] = kwargs.get("input")
        captured["env"] = kwargs.get("env")
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )

    return captured, fake_run
