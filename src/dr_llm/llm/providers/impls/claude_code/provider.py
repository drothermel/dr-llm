from __future__ import annotations

import json
import os
from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import HeadlessExecutionError
from dr_llm.llm.names import EffortSpec, ProviderName
from dr_llm.llm.providers.names import ApiKeyNames
from dr_llm.llm.providers.transports.headless_base import (
    BaseHeadlessProvider,
    HEADLESS_DEFAULT_EMPTY_PROMPT,
    HeadlessControlResult,
    HeadlessRequestPayload,
    ParsedHeadlessOutput,
    messages_to_prompt,
)
from dr_llm.llm.providers.impls.claude_code.controls import (
    ClaudeHeadlessControlMapping,
)
from dr_llm.llm.providers.impls.claude_code.families import (
    ClaudeCodeModelFamily,
)
from dr_llm.llm.providers.transports.headless_config import (
    ClaudeCodeProviderConfig,
)
from dr_llm.llm.request import LlmRequest


CLAUDE_DEFAULT_COMMAND = [
    "claude",
    "-p",
    "--output-format",
    "json",
    "--input-format",
    "text",
    "--system-prompt",
    "",
    "--tools",
    "",
    "--disable-slash-commands",
    "--no-session-persistence",
    "--setting-sources",
    "user",
]
ANTHROPIC_BASE_URL_ENV = "ANTHROPIC_BASE_URL"
ANTHROPIC_AUTH_TOKEN_ENV = "ANTHROPIC_AUTH_TOKEN"


class ClaudeCodeUrls(StrEnum):
    MODELS_DOCS = "https://code.claude.com/docs/en/model-config"


class ClaudeHeadlessUsage(BaseModel):
    model_config = ConfigDict(extra="allow")

    input_tokens: int | None = None
    output_tokens: int | None = None


class ClaudeHeadlessRawResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    KEY_RESULT: ClassVar[str] = "result"
    KEY_STOP_REASON: ClassVar[str] = "stop_reason"
    KEY_TOTAL_COST_USD: ClassVar[str] = "total_cost_usd"

    type: str | None = None
    subtype: str | None = None
    is_error: bool = False
    result: str | None = None
    usage: ClaudeHeadlessUsage | None = None
    stop_reason: str | None = None
    total_cost_usd: float | None = None


class ClaudeHeadlessResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stdout_text: str = Field(exclude=True)
    stderr: str = Field(exclude=True)
    body: ClaudeHeadlessRawResponse | None = None

    @classmethod
    def from_stdout(
        cls, *, stdout: str, stderr: str
    ) -> ClaudeHeadlessResponse:
        stdout_clean = stdout.strip()
        if not stdout_clean:
            return cls(stdout_text="", stderr=stderr)

        try:
            body = json.loads(stdout_clean)
        except json.JSONDecodeError:
            return cls(stdout_text=stdout_clean, stderr=stderr)
        if not isinstance(body, dict):
            return cls(stdout_text=stdout_clean, stderr=stderr)
        return cls(
            stdout_text=stdout_clean,
            stderr=stderr,
            body=ClaudeHeadlessRawResponse(**body),
        )

    def to_parsed_output(self, *, provider_name: str) -> ParsedHeadlessOutput:
        if not self.stdout_text:
            return ParsedHeadlessOutput.empty(stderr=self.stderr)
        if self.body is None:
            return ParsedHeadlessOutput.text_response(
                text=self.stdout_text,
                stderr=self.stderr,
            )
        if self.body.is_error:
            raise HeadlessExecutionError(
                f"{provider_name} command returned error: {self.body.result}"
            )

        usage = self.body.usage or ClaudeHeadlessUsage()
        normalized_body: dict[str, Any] = {
            "text": str(self.body.result or ""),
            "finish_reason": self.body.stop_reason or "stop",
            "usage": {
                key: value
                for key, value in {
                    "prompt_tokens": usage.input_tokens,
                    "completion_tokens": usage.output_tokens,
                }.items()
                if value is not None
            },
        }
        if self.body.total_cost_usd is not None:
            normalized_body["total_cost"] = self.body.total_cost_usd
            normalized_body["currency"] = "USD"

        return ParsedHeadlessOutput.from_body(
            body=normalized_body,
            raw_json=self.body.model_dump(mode="json", exclude_none=True),
        )


class ClaudeCodeProvider(BaseHeadlessProvider):
    _config: ClaudeCodeProviderConfig

    def __init__(
        self,
        command: list[str] | None = None,
        *,
        name: str = ProviderName.CLAUDE_CODE,
        env_overrides: dict[str, str] | None = None,
        api_key_env: str | None = None,
    ) -> None:
        super().__init__(
            config=ClaudeCodeProviderConfig(
                name=name,
                command=command or CLAUDE_DEFAULT_COMMAND,
                env_overrides=env_overrides or {},
                api_key_env=api_key_env,
            ),
        )

    def subprocess_env(
        self,
        request: LlmRequest,
        payload: HeadlessRequestPayload,
    ) -> dict[str, str]:
        del request, payload
        env = {**os.environ, **self._config.env_overrides}
        if self._config.api_key_env is not None:
            key_value = os.getenv(self._config.api_key_env)
            if key_value:
                env[ANTHROPIC_AUTH_TOKEN_ENV] = key_value
                env[ApiKeyNames.ANTHROPIC] = key_value
        return env

    def command_for_request(
        self,
        request: LlmRequest,
        payload: HeadlessRequestPayload,
        control_mapping: HeadlessControlResult,
    ) -> list[str]:
        del payload
        if (
            self.name == ProviderName.CLAUDE_CODE
            and not ClaudeCodeModelFamily.CLAUDE.in_family(request.model)
        ):
            raise HeadlessExecutionError(
                f"{ProviderName.CLAUDE_CODE} requires canonical model ids like 'claude-sonnet-4-6'"
            )
        command = [*self._config.command]
        command.extend(["--model", request.model])
        if request.effort != EffortSpec.NA:
            command.extend(["--effort", request.effort])
        elif control_mapping.cli_args:
            command.extend(control_mapping.cli_args)
        return command

    def control_mapping(self, request: LlmRequest) -> HeadlessControlResult:
        return ClaudeHeadlessControlMapping.from_base(request.reasoning)

    def stdin_for_request(
        self,
        request: LlmRequest,
        payload: HeadlessRequestPayload,
    ) -> str:
        del payload
        prompt = messages_to_prompt(request.messages).strip()
        return prompt or HEADLESS_DEFAULT_EMPTY_PROMPT

    def parse_stdout(
        self,
        *,
        request: LlmRequest,
        stdout: str,
        stderr: str,
    ) -> ParsedHeadlessOutput:
        del request
        return ClaudeHeadlessResponse.from_stdout(
            stdout=stdout,
            stderr=stderr,
        ).to_parsed_output(provider_name=self.name)
