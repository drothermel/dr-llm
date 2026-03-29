from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import HeadlessExecutionError
from dr_llm.generation.models import (
    CallMode,
    LlmRequest,
    LlmResponse,
    Message,
    ReasoningConfig,
    TokenUsage,
)
from dr_llm.logging import emit_generation_event
from dr_llm.providers.headless.config import HeadlessProviderConfig
from dr_llm.providers.provider_adapter import ProviderAdapter
from dr_llm.providers.utils import (
    parse_cost_info,
    parse_reasoning,
    parse_reasoning_tokens,
)
from dr_llm.reasoning import ReasoningMappingResult


HEADLESS_DEFAULT_EMPTY_PROMPT = " "
_DISALLOWED_HEADLESS_COMMANDS = {"sh", "bash", "zsh", "fish", "pwsh", "powershell"}


class HeadlessRequestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    messages: list[Message]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    reasoning: ReasoningConfig | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_llm_request(cls, request: LlmRequest) -> HeadlessRequestPayload:
        return cls(
            provider=request.provider,
            model=request.model,
            messages=request.messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            reasoning=request.reasoning,
            metadata=request.metadata,
        )

    def json_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_computed_fields=True)


class HeadlessUsagePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None


class ParsedHeadlessOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = ""
    finish_reason: str | None = None
    usage: HeadlessUsagePayload = Field(default_factory=HeadlessUsagePayload)
    body: dict[str, Any] | None = Field(default=None, exclude=True, repr=False)
    raw_json: dict[str, Any]

    @classmethod
    def empty(cls, *, stderr: str) -> ParsedHeadlessOutput:
        return cls(raw_json={"stdout": "", "stderr": stderr})

    @classmethod
    def text_response(cls, *, text: str, stderr: str) -> ParsedHeadlessOutput:
        return cls(text=text, raw_json={"stdout": text, "stderr": stderr})

    @classmethod
    def from_body(
        cls,
        *,
        body: dict[str, Any],
        raw_json: dict[str, Any] | None = None,
    ) -> ParsedHeadlessOutput:
        raw_usage_raw = body.get("usage")
        raw_usage: dict[str, Any] = (
            raw_usage_raw if isinstance(raw_usage_raw, dict) else {}
        )
        reasoning_tokens = parse_reasoning_tokens(raw_usage)
        prompt_tokens = raw_usage.get("prompt_tokens")
        completion_tokens = raw_usage.get("completion_tokens")
        total_tokens = raw_usage.get("total_tokens")
        return cls(
            text=str(body.get("text") or ""),
            finish_reason=body.get("finish_reason")
            if isinstance(body.get("finish_reason"), str)
            else None,
            usage=HeadlessUsagePayload(
                prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
                completion_tokens=(
                    completion_tokens if isinstance(completion_tokens, int) else None
                ),
                total_tokens=total_tokens if isinstance(total_tokens, int) else None,
                reasoning_tokens=reasoning_tokens,
            ),
            body=body,
            raw_json=raw_json or body,
        )

    def to_llm_response(self, request: LlmRequest, *, latency_ms: int) -> LlmResponse:
        body = self.body or {}
        message_raw = (
            body.get("message")
            if isinstance(body.get("message"), dict)
            else body
            if body
            else None
        )
        reasoning, reasoning_details = parse_reasoning(
            message_raw if isinstance(message_raw, dict) else None
        )
        if reasoning is None and isinstance(body.get("reasoning"), str):
            reasoning = body.get("reasoning")
        if reasoning_details is None and isinstance(
            body.get("reasoning_details"), list
        ):
            raw_reasoning_details = body.get("reasoning_details")
            if isinstance(raw_reasoning_details, list):
                reasoning_details = [
                    item for item in raw_reasoning_details if isinstance(item, dict)
                ]
        return LlmResponse(
            text=self.text,
            finish_reason=self.finish_reason,
            usage=TokenUsage.from_raw(
                prompt_tokens=self.usage.prompt_tokens,
                completion_tokens=self.usage.completion_tokens,
                total_tokens=self.usage.total_tokens,
                reasoning_tokens=self.usage.reasoning_tokens,
            ),
            reasoning=reasoning,
            reasoning_details=reasoning_details,
            cost=parse_cost_info(body),
            raw_json=self.raw_json,
            latency_ms=latency_ms,
            provider=request.provider,
            model=request.model,
            mode=CallMode.headless,
        )


def messages_to_prompt(messages: list[Message]) -> str:
    parts: list[str] = []
    for message in messages:
        content = message.content.strip()
        if not content:
            continue
        parts.append(f"{message.role}: {content}")
    return "\n\n".join(parts)


def validate_headless_command(command: list[str]) -> None:
    if not command:
        raise HeadlessExecutionError("headless command must be non-empty")
    executable = command[0].strip()
    if not executable:
        raise HeadlessExecutionError("headless command executable must be non-empty")
    command_name = os.path.basename(executable).lower()
    if command_name in _DISALLOWED_HEADLESS_COMMANDS:
        raise HeadlessExecutionError(
            f"headless command executable {command_name!r} is disallowed"
        )


def sanitize_io_for_logs(
    value: str,
    *,
    log_full_io: bool,
    redact_io: bool,
    max_logged_chars: int,
) -> str:
    if not log_full_io:
        return "<omitted>"
    sanitized = value
    if redact_io:
        sanitized = f"<redacted len={len(value)}>"
    if len(sanitized) > max_logged_chars:
        return f"{sanitized[:max_logged_chars]}...[truncated]"
    return sanitized


class BaseHeadlessAdapter(ProviderAdapter):
    _config: HeadlessProviderConfig

    def __init__(self, *, config: HeadlessProviderConfig) -> None:
        self._config = config

    def payload_for_request(self, request: LlmRequest) -> HeadlessRequestPayload:
        return HeadlessRequestPayload.from_llm_request(request)

    def command_for_request(
        self,
        request: LlmRequest,
        payload: HeadlessRequestPayload,
        reasoning_mapping: ReasoningMappingResult,
    ) -> list[str]:
        return [*self._config.command]

    def stdin_for_request(
        self,
        request: LlmRequest,
        payload: HeadlessRequestPayload,
    ) -> str:
        return json.dumps(payload.json_payload(), ensure_ascii=True)

    def subprocess_env(
        self,
        request: LlmRequest,
        payload: HeadlessRequestPayload,
    ) -> dict[str, str]:
        del request, payload
        return {**os.environ, **self._config.env_overrides}

    def reasoning_mapping(self, request: LlmRequest) -> ReasoningMappingResult:
        return ReasoningMappingResult()

    def parse_stdout(
        self,
        *,
        request: LlmRequest,
        stdout: str,
        stderr: str,
    ) -> ParsedHeadlessOutput:
        del request
        stdout_clean = stdout.strip()
        if not stdout_clean:
            return ParsedHeadlessOutput.empty(stderr=stderr)
        try:
            body = json.loads(stdout_clean)
        except json.JSONDecodeError:
            return ParsedHeadlessOutput.text_response(text=stdout_clean, stderr=stderr)
        if not isinstance(body, dict):
            return ParsedHeadlessOutput.text_response(text=stdout_clean, stderr=stderr)
        return ParsedHeadlessOutput.from_body(body=body, raw_json=body)

    def generate(self, request: LlmRequest) -> LlmResponse:
        payload = self.payload_for_request(request)
        reasoning_mapping = self.reasoning_mapping(request)
        command = self.command_for_request(request, payload, reasoning_mapping)
        validate_headless_command(command)
        stdin_text = self.stdin_for_request(request, payload)
        started = time.perf_counter()
        logged_stdin = sanitize_io_for_logs(
            stdin_text,
            log_full_io=self._config.log_full_io,
            redact_io=self._config.redact_io,
            max_logged_chars=self._config.max_logged_chars,
        )
        try:
            proc = subprocess.run(
                command,
                input=stdin_text,
                text=True,
                capture_output=True,
                timeout=self._config.timeout_seconds,
                env=self.subprocess_env(request, payload),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            emit_generation_event(
                event_type="provider.raw_response",
                stage=f"{self.name}.subprocess_timeout",
                payload={
                    "command": command,
                    "stdin": logged_stdin,
                    "timeout_seconds": self._config.timeout_seconds,
                },
            )
            raise HeadlessExecutionError(
                f"{self.name} command timed out after {self._config.timeout_seconds}s"
            ) from exc

        latency_ms = int((time.perf_counter() - started) * 1000)
        logged_stdout = sanitize_io_for_logs(
            proc.stdout,
            log_full_io=self._config.log_full_io,
            redact_io=self._config.redact_io,
            max_logged_chars=self._config.max_logged_chars,
        )
        logged_stderr = sanitize_io_for_logs(
            proc.stderr,
            log_full_io=self._config.log_full_io,
            redact_io=self._config.redact_io,
            max_logged_chars=self._config.max_logged_chars,
        )
        emit_generation_event(
            event_type="provider.raw_response",
            stage=f"{self.name}.subprocess_result",
            payload={
                "command": command,
                "stdin": logged_stdin,
                "returncode": proc.returncode,
                "stdout": logged_stdout,
                "stderr": logged_stderr,
            },
        )
        if proc.returncode != 0:
            raise HeadlessExecutionError(
                f"{self.name} command failed with exit code {proc.returncode}: {logged_stderr[:800]}"
            )
        response = self.parse_stdout(
            request=request,
            stdout=proc.stdout,
            stderr=proc.stderr,
        ).to_llm_response(request, latency_ms=latency_ms)
        if reasoning_mapping.warnings:
            response = response.model_copy(
                update={"warnings": [*response.warnings, *reasoning_mapping.warnings]}
            )
        return response
