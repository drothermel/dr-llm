from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from typing import Any

from pydantic import BaseModel, Field

from dr_llm.errors import HeadlessExecutionError
from dr_llm.logging import emit_generation_event
from dr_llm.providers.headless_provider_config import (
    ClaudeHeadlessProviderConfig,
    HeadlessProviderConfig,
)
from dr_llm.providers.provider_adapter import (
    ProviderAdapter,
)
from dr_llm.providers.utils import (
    parse_cost_info,
    parse_reasoning,
    parse_reasoning_tokens,
)
from dr_llm.reasoning import (
    ReasoningMappingResult,
    map_reasoning_for_claude_headless,
    map_reasoning_for_codex_headless,
)
from dr_llm.generation.models import (
    CallMode,
    LlmRequest,
    LlmResponse,
    Message,
    ReasoningConfig,
    TokenUsage,
)


class _HeadlessRequestPayload(BaseModel):
    provider: str
    model: str
    messages: list[Message]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    reasoning: ReasoningConfig | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class _HeadlessUsagePayload(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None


CODEX_DEFAULT_COMMAND = [
    "codex",
    "exec",
    "--json",
    "--skip-git-repo-check",
    "--sandbox",
    "read-only",
    "--disable",
    "web_search_request",
    "-c",
    "include_plan_tool=false",
    "-c",
    "project_doc_max_bytes=0",
]
CLAUDE_DEFAULT_COMMAND = [
    "claude",
    "-p",
    "--output-format",
    "json",
    "--input-format",
    "text",
    "--system-prompt",
    "",
    "--disable-slash-commands",
    "--no-session-persistence",
    "--setting-sources",
    "user",
]

CODEX_PROMPT_SENTINEL = "-"
CODEX_NEUTRAL_INSTRUCTIONS_CONTENT = "."
HEADLESS_DEFAULT_EMPTY_PROMPT = " "
ANTHROPIC_BASE_URL_ENV = "ANTHROPIC_BASE_URL"
ANTHROPIC_AUTH_TOKEN_ENV = "ANTHROPIC_AUTH_TOKEN"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
MINIMAX_ANTHROPIC_BASE_URL = "https://api.minimax.io/anthropic"
KIMI_CODING_BASE_URL = "https://api.kimi.com/coding/"
MINIMAX_API_KEY_ENV = "MINIMAX_API_KEY"
KIMI_API_KEY_ENV = "KIMI_API_KEY"
CLAUDE_CANONICAL_MODEL_PREFIX = "claude-"
KEY_TYPE = "type"
KEY_ITEM = "item"
KEY_TEXT = "text"
KEY_USAGE = "usage"
KEY_TURN_COMPLETED = "turn.completed"
KEY_ITEM_COMPLETED = "item.completed"
KEY_AGENT_MESSAGE = "agent_message"
KEY_RESULT = "result"
KEY_STOP_REASON = "stop_reason"
KEY_TOTAL_COST_USD = "total_cost_usd"
KEY_IS_ERROR = "is_error"
KEY_INPUT_TOKENS = "input_tokens"
KEY_OUTPUT_TOKENS = "output_tokens"
KEY_NON_JSON_STDOUT_LINES = "non_json_stdout_lines"

logger = logging.getLogger(__name__)

_DISALLOWED_HEADLESS_COMMANDS = {"sh", "bash", "zsh", "fish", "pwsh", "powershell"}


def _message_label(message: Message) -> str:
    return message.role


def _messages_to_prompt(messages: list[Message]) -> str:
    parts: list[str] = []
    for message in messages:
        content = message.content.strip()
        if not content:
            continue
        parts.append(f"{_message_label(message)}: {content}")
    return "\n\n".join(parts)


def _validate_headless_command(command: list[str]) -> None:
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


def _sanitize_io_for_logs(
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


class _BaseHeadlessAdapter(ProviderAdapter):
    _config: HeadlessProviderConfig

    def __init__(self, *, config: HeadlessProviderConfig) -> None:
        self._config = config

    def _payload(self, request: LlmRequest) -> dict[str, Any]:
        return _HeadlessRequestPayload(
            provider=request.provider,
            model=request.model,
            messages=request.messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            reasoning=request.reasoning,
            metadata=request.metadata,
        ).model_dump(
            mode="json",
            exclude_computed_fields=True,
        )

    def _command_for_request(
        self,
        request: LlmRequest,
        payload: dict[str, Any],
        reasoning_mapping: ReasoningMappingResult,
    ) -> list[str]:
        return [*self._config.command]

    def _stdin_for_request(self, request: LlmRequest, payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=True)

    def _subprocess_env(
        self, request: LlmRequest, payload: dict[str, Any]
    ) -> dict[str, str]:
        return {**os.environ, **self._config.env_overrides}

    def _reasoning_mapping(self, request: LlmRequest) -> ReasoningMappingResult:
        return ReasoningMappingResult()

    def _empty_response(
        self,
        *,
        request: LlmRequest,
        latency_ms: int,
        stderr: str,
    ) -> LlmResponse:
        return LlmResponse(
            text="",
            finish_reason=None,
            usage=TokenUsage(),
            raw_json={"stdout": "", "stderr": stderr},
            latency_ms=latency_ms,
            provider=request.provider,
            model=request.model,
            mode=CallMode.headless,
        )

    def _text_response(
        self,
        *,
        request: LlmRequest,
        latency_ms: int,
        stderr: str,
        text: str,
    ) -> LlmResponse:
        return LlmResponse(
            text=text,
            finish_reason=None,
            usage=TokenUsage(),
            raw_json={"stdout": text, "stderr": stderr},
            latency_ms=latency_ms,
            provider=request.provider,
            model=request.model,
            mode=CallMode.headless,
        )

    def _response_from_body(
        self,
        *,
        request: LlmRequest,
        latency_ms: int,
        body: Any,
        raw_json: dict[str, Any] | None = None,
    ) -> LlmResponse:
        raw_usage = body.get(KEY_USAGE) if isinstance(body, dict) else None
        reasoning_tokens = parse_reasoning_tokens(
            raw_usage if isinstance(raw_usage, dict) else {}
        )
        usage = TokenUsage.from_raw(
            prompt_tokens=(raw_usage or {}).get("prompt_tokens")
            if isinstance(raw_usage, dict)
            else None,
            completion_tokens=(raw_usage or {}).get("completion_tokens")
            if isinstance(raw_usage, dict)
            else None,
            total_tokens=(raw_usage or {}).get("total_tokens")
            if isinstance(raw_usage, dict)
            else None,
            reasoning_tokens=reasoning_tokens,
        )
        message_raw = (
            body.get("message")
            if isinstance(body, dict) and isinstance(body.get("message"), dict)
            else body
        )
        reasoning, reasoning_details = parse_reasoning(
            message_raw if isinstance(message_raw, dict) else None
        )
        if (
            reasoning is None
            and isinstance(body, dict)
            and isinstance(body.get("reasoning"), str)
        ):
            reasoning = body.get("reasoning")
        if (
            reasoning_details is None
            and isinstance(body, dict)
            and isinstance(body.get("reasoning_details"), list)
        ):
            raw_reasoning_details = body.get("reasoning_details")
            if isinstance(raw_reasoning_details, list):
                reasoning_details = [
                    item for item in raw_reasoning_details if isinstance(item, dict)
                ]
        cost = parse_cost_info(body if isinstance(body, dict) else {})
        text = str(body.get(KEY_TEXT) or "") if isinstance(body, dict) else str(body)
        finish_reason = body.get("finish_reason") if isinstance(body, dict) else None
        resolved_raw_json = (
            raw_json
            if raw_json is not None
            else body
            if isinstance(body, dict)
            else {"body": body}
        )

        return LlmResponse(
            text=text,
            finish_reason=finish_reason,
            usage=usage,
            reasoning=reasoning,
            reasoning_details=reasoning_details,
            cost=cost,
            raw_json=resolved_raw_json,
            latency_ms=latency_ms,
            provider=request.provider,
            model=request.model,
            mode=CallMode.headless,
        )

    def _parse_stdout(
        self,
        *,
        request: LlmRequest,
        latency_ms: int,
        stdout: str,
        stderr: str,
    ) -> LlmResponse:
        stdout_clean = stdout.strip()
        if not stdout_clean:
            return self._empty_response(
                request=request,
                latency_ms=latency_ms,
                stderr=stderr,
            )
        try:
            body = json.loads(stdout_clean)
        except json.JSONDecodeError:
            return self._text_response(
                request=request,
                latency_ms=latency_ms,
                stderr=stderr,
                text=stdout_clean,
            )
        if not isinstance(body, dict):
            return self._text_response(
                request=request,
                latency_ms=latency_ms,
                stderr=stderr,
                text=stdout_clean,
            )
        return self._response_from_body(
            request=request,
            latency_ms=latency_ms,
            body=body,
            raw_json=body,
        )

    def generate(self, request: LlmRequest) -> LlmResponse:
        payload = self._payload(request)
        reasoning_mapping = self._reasoning_mapping(request)
        command = self._command_for_request(request, payload, reasoning_mapping)
        _validate_headless_command(command)
        stdin_text = self._stdin_for_request(request, payload)
        started = time.perf_counter()
        logged_stdin = _sanitize_io_for_logs(
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
                env=self._subprocess_env(request, payload),
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
        logged_stdout = _sanitize_io_for_logs(
            proc.stdout,
            log_full_io=self._config.log_full_io,
            redact_io=self._config.redact_io,
            max_logged_chars=self._config.max_logged_chars,
        )
        logged_stderr = _sanitize_io_for_logs(
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
        response = self._parse_stdout(
            request=request,
            latency_ms=latency_ms,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
        if reasoning_mapping.warnings:
            response = response.model_copy(
                update={"warnings": [*response.warnings, *reasoning_mapping.warnings]}
            )
        return response


class CodexHeadlessAdapter(_BaseHeadlessAdapter):
    def __init__(self, command: list[str] | None = None) -> None:
        super().__init__(
            config=HeadlessProviderConfig(
                name="codex",
                command=command or CODEX_DEFAULT_COMMAND,
            ),
        )
        self._neutral_instructions_file: str | None = None

    def _ensure_neutral_instructions_file(self) -> str:
        if self._neutral_instructions_file is not None:
            return self._neutral_instructions_file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".md",
            prefix="dr-llm-codex-neutral-",
            delete=False,
            encoding="utf-8",
        ) as file:
            file.write(CODEX_NEUTRAL_INSTRUCTIONS_CONTENT)
            self._neutral_instructions_file = file.name
        return self._neutral_instructions_file

    def _command_for_request(
        self,
        request: LlmRequest,
        payload: dict[str, Any],
        reasoning_mapping: ReasoningMappingResult,
    ) -> list[str]:
        command = [*self._config.command]
        command.extend(["-m", request.model])
        instructions_path = self._ensure_neutral_instructions_file()
        command.extend(
            [
                "-c",
                f'model_instructions_file="{instructions_path}"',
                CODEX_PROMPT_SENTINEL,
            ]
        )
        return command

    def _reasoning_mapping(self, request: LlmRequest) -> ReasoningMappingResult:
        return map_reasoning_for_codex_headless(
            request.reasoning,
            provider=request.provider,
            mode=CallMode.headless,
        )

    def _stdin_for_request(self, request: LlmRequest, payload: dict[str, Any]) -> str:
        prompt = _messages_to_prompt(request.messages).strip()
        return prompt or HEADLESS_DEFAULT_EMPTY_PROMPT

    def _parse_stdout(
        self,
        *,
        request: LlmRequest,
        latency_ms: int,
        stdout: str,
        stderr: str,
    ) -> LlmResponse:
        stdout_clean = stdout.strip()
        if not stdout_clean:
            return self._empty_response(
                request=request,
                latency_ms=latency_ms,
                stderr=stderr,
            )
        try:
            parsed_body = json.loads(stdout_clean)
        except json.JSONDecodeError:
            parsed_body = None
        if isinstance(parsed_body, dict) and KEY_TYPE not in parsed_body:
            return self._response_from_body(
                request=request,
                latency_ms=latency_ms,
                body=parsed_body,
                raw_json=parsed_body,
            )

        lines = stdout.splitlines()
        events: list[dict[str, Any]] = []
        passthrough_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                passthrough_lines.append(stripped)
                continue
            if isinstance(parsed, dict):
                events.append(parsed)
            else:
                passthrough_lines.append(stripped)

        if not events:
            return self._text_response(
                request=request,
                latency_ms=latency_ms,
                stderr=stderr,
                text=stdout_clean,
            )

        for event in events:
            if event.get(KEY_TYPE) == "error":
                logger.debug(
                    "headless error event provider=%s event=%s",
                    self.name,
                    event,
                )
                event_message = event.get("message") or event.get("code") or event
                raise HeadlessExecutionError(
                    f"{self.name} command returned error event: {str(event_message)[:200]}"
                )

        body = _codex_events_to_body(events, passthrough_lines=passthrough_lines)
        raw_json = {
            "events": events,
            KEY_NON_JSON_STDOUT_LINES: passthrough_lines,
            "stderr": stderr,
        }
        return self._response_from_body(
            request=request,
            latency_ms=latency_ms,
            body=body,
            raw_json=raw_json,
        )


class ClaudeHeadlessAdapter(_BaseHeadlessAdapter):
    _config: ClaudeHeadlessProviderConfig

    def __init__(
        self,
        command: list[str] | None = None,
        *,
        name: str = "claude-code",
        env_overrides: dict[str, str] | None = None,
        api_key_env: str | None = None,
    ) -> None:
        super().__init__(
            config=ClaudeHeadlessProviderConfig(
                name=name,
                command=command or CLAUDE_DEFAULT_COMMAND,
                env_overrides=env_overrides or {},
                api_key_env=api_key_env,
            ),
        )

    def _subprocess_env(
        self, request: LlmRequest, payload: dict[str, Any]
    ) -> dict[str, str]:
        env = super()._subprocess_env(request, payload)
        if self._config.api_key_env is not None:
            key_value = os.getenv(self._config.api_key_env)
            if key_value:
                env[ANTHROPIC_AUTH_TOKEN_ENV] = key_value
                env[ANTHROPIC_API_KEY_ENV] = key_value
        return env

    def _command_for_request(
        self,
        request: LlmRequest,
        payload: dict[str, Any],
        reasoning_mapping: ReasoningMappingResult,
    ) -> list[str]:
        if self.name == "claude-code" and not request.model.startswith(
            CLAUDE_CANONICAL_MODEL_PREFIX
        ):
            raise HeadlessExecutionError(
                "claude-code requires canonical model ids like 'claude-sonnet-4-6'"
            )
        command = [*self._config.command]
        command.extend(["--model", request.model])
        if reasoning_mapping.cli_args:
            command.extend(reasoning_mapping.cli_args)
        return command

    def _reasoning_mapping(self, request: LlmRequest) -> ReasoningMappingResult:
        return map_reasoning_for_claude_headless(
            request.reasoning,
            provider=request.provider,
            mode=CallMode.headless,
        )

    def _stdin_for_request(self, request: LlmRequest, payload: dict[str, Any]) -> str:
        prompt = _messages_to_prompt(request.messages).strip()
        return prompt or HEADLESS_DEFAULT_EMPTY_PROMPT

    def _parse_stdout(
        self,
        *,
        request: LlmRequest,
        latency_ms: int,
        stdout: str,
        stderr: str,
    ) -> LlmResponse:
        stdout_clean = stdout.strip()
        if not stdout_clean:
            return self._empty_response(
                request=request,
                latency_ms=latency_ms,
                stderr=stderr,
            )

        try:
            body = json.loads(stdout_clean)
        except json.JSONDecodeError:
            return self._text_response(
                request=request,
                latency_ms=latency_ms,
                stderr=stderr,
                text=stdout_clean,
            )
        if not isinstance(body, dict):
            return self._text_response(
                request=request,
                latency_ms=latency_ms,
                stderr=stderr,
                text=stdout_clean,
            )
        if body.get(KEY_IS_ERROR) is True:
            raise HeadlessExecutionError(
                f"{self.name} command returned error: {body.get(KEY_RESULT)}"
            )

        usage = body.get(KEY_USAGE) if isinstance(body.get(KEY_USAGE), dict) else {}
        prompt_tokens = usage.get(KEY_INPUT_TOKENS)
        completion_tokens = usage.get(KEY_OUTPUT_TOKENS)
        usage_payload = _HeadlessUsagePayload(
            prompt_tokens=int(prompt_tokens)
            if isinstance(prompt_tokens, int)
            else None,
            completion_tokens=int(completion_tokens)
            if isinstance(completion_tokens, int)
            else None,
        )
        normalized_body: dict[str, Any] = {
            KEY_TEXT: str(body.get(KEY_RESULT) or ""),
            "finish_reason": body.get(KEY_STOP_REASON) or "stop",
            KEY_USAGE: usage_payload.model_dump(mode="json", exclude_none=True),
        }
        if isinstance(body.get(KEY_TOTAL_COST_USD), (int, float)):
            normalized_body["total_cost"] = float(body.get(KEY_TOTAL_COST_USD))
            normalized_body["currency"] = "USD"

        return self._response_from_body(
            request=request,
            latency_ms=latency_ms,
            body=normalized_body,
            raw_json=body,
        )


class ClaudeHeadlessMiniMaxAdapter(ClaudeHeadlessAdapter):
    def __init__(self, command: list[str] | None = None) -> None:
        super().__init__(
            command=command,
            name="claude-code-minimax",
            env_overrides={ANTHROPIC_BASE_URL_ENV: MINIMAX_ANTHROPIC_BASE_URL},
            api_key_env=MINIMAX_API_KEY_ENV,
        )


class ClaudeHeadlessKimiAdapter(ClaudeHeadlessAdapter):
    def __init__(self, command: list[str] | None = None) -> None:
        super().__init__(
            command=command,
            name="claude-code-kimi",
            env_overrides={ANTHROPIC_BASE_URL_ENV: KIMI_CODING_BASE_URL},
            api_key_env=KIMI_API_KEY_ENV,
        )


def _codex_events_to_body(
    events: list[dict[str, Any]],
    *,
    passthrough_lines: list[str],
) -> dict[str, Any]:
    text_chunks: list[str] = []
    usage: dict[str, int] = {}
    finish_reason: str | None = None

    for event in events:
        event_type = event.get(KEY_TYPE)
        if event_type == KEY_ITEM_COMPLETED:
            item = event.get(KEY_ITEM)
            if not isinstance(item, dict):
                continue
            item_type = item.get(KEY_TYPE)
            if item_type == KEY_AGENT_MESSAGE:
                text = item.get(KEY_TEXT)
                if isinstance(text, str) and text:
                    text_chunks.append(text)
        if event_type == KEY_TURN_COMPLETED:
            finish_reason = "stop"
            usage_raw = event.get(KEY_USAGE)
            if isinstance(usage_raw, dict):
                prompt_tokens = usage_raw.get(KEY_INPUT_TOKENS)
                completion_tokens = usage_raw.get(KEY_OUTPUT_TOKENS)
                usage_payload = _HeadlessUsagePayload(
                    prompt_tokens=prompt_tokens
                    if isinstance(prompt_tokens, int)
                    else None,
                    completion_tokens=completion_tokens
                    if isinstance(completion_tokens, int)
                    else None,
                    total_tokens=(
                        prompt_tokens + completion_tokens
                        if isinstance(prompt_tokens, int)
                        and isinstance(completion_tokens, int)
                        else None
                    ),
                )
                usage = usage_payload.model_dump(mode="json", exclude_none=True)

    body: dict[str, Any] = {
        KEY_TEXT: "\n".join(text_chunks).strip(),
        "finish_reason": finish_reason,
        KEY_USAGE: usage,
    }
    if passthrough_lines:
        body[KEY_NON_JSON_STDOUT_LINES] = passthrough_lines
    return body
