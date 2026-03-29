from __future__ import annotations

import json
import logging
import tempfile
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import HeadlessExecutionError
from dr_llm.generation.models import CallMode, LlmRequest
from dr_llm.providers.headless.base import (
    BaseHeadlessAdapter,
    HEADLESS_DEFAULT_EMPTY_PROMPT,
    HeadlessRequestPayload,
    ParsedHeadlessOutput,
    messages_to_prompt,
)
from dr_llm.providers.headless.config import HeadlessProviderConfig
from dr_llm.reasoning import ReasoningMappingResult, map_reasoning_for_codex_headless


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
CODEX_PROMPT_SENTINEL = "-"
CODEX_NEUTRAL_INSTRUCTIONS_CONTENT = "."

logger = logging.getLogger(__name__)


class CodexEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    TYPE_ERROR: ClassVar[str] = "error"
    TYPE_ITEM_COMPLETED: ClassVar[str] = "item.completed"
    TYPE_TURN_COMPLETED: ClassVar[str] = "turn.completed"
    ITEM_AGENT_MESSAGE: ClassVar[str] = "agent_message"

    type: str
    item: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    message: str | None = None
    code: str | None = None

    def is_error(self) -> bool:
        return self.type == self.TYPE_ERROR

    def agent_text(self) -> str | None:
        if self.type != self.TYPE_ITEM_COMPLETED or not isinstance(self.item, dict):
            return None
        if self.item.get("type") != self.ITEM_AGENT_MESSAGE:
            return None
        text = self.item.get("text")
        return text if isinstance(text, str) and text else None

    def usage_payload(self) -> dict[str, int] | None:
        if self.type != self.TYPE_TURN_COMPLETED or not isinstance(self.usage, dict):
            return None
        prompt_tokens = self.usage.get("input_tokens")
        completion_tokens = self.usage.get("output_tokens")
        usage: dict[str, int] = {}
        if isinstance(prompt_tokens, int):
            usage["prompt_tokens"] = prompt_tokens
        if isinstance(completion_tokens, int):
            usage["completion_tokens"] = completion_tokens
        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            usage["total_tokens"] = prompt_tokens + completion_tokens
        return usage

    def error_message(self) -> str:
        return str(self.message or self.code or self.model_dump(mode="json"))


class CodexHeadlessResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stdout_text: str = Field(exclude=True)
    stderr: str = Field(exclude=True)
    direct_body: dict[str, Any] | None = None
    events: list[CodexEvent] = Field(default_factory=list)
    passthrough_lines: list[str] = Field(default_factory=list)

    @classmethod
    def from_stdout(cls, *, stdout: str, stderr: str) -> CodexHeadlessResponse:
        stdout_clean = stdout.strip()
        if not stdout_clean:
            return cls(stdout_text="", stderr=stderr)

        try:
            parsed_body = json.loads(stdout_clean)
        except json.JSONDecodeError:
            parsed_body = None
        if isinstance(parsed_body, dict) and "type" not in parsed_body:
            return cls(stdout_text=stdout_clean, stderr=stderr, direct_body=parsed_body)

        events: list[CodexEvent] = []
        passthrough_lines: list[str] = []
        for line in stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed_line = json.loads(stripped)
            except json.JSONDecodeError:
                passthrough_lines.append(stripped)
                continue
            if not isinstance(parsed_line, dict):
                passthrough_lines.append(stripped)
                continue
            events.append(CodexEvent(**parsed_line))

        return cls(
            stdout_text=stdout_clean,
            stderr=stderr,
            events=events,
            passthrough_lines=passthrough_lines,
        )

    def to_parsed_output(self, *, provider_name: str) -> ParsedHeadlessOutput:
        if not self.stdout_text:
            return ParsedHeadlessOutput.empty(stderr=self.stderr)
        if self.direct_body is not None:
            return ParsedHeadlessOutput.from_body(
                body=self.direct_body,
                raw_json=self.direct_body,
            )
        if not self.events:
            return ParsedHeadlessOutput.text_response(
                text=self.stdout_text,
                stderr=self.stderr,
            )

        for event in self.events:
            if event.is_error():
                logger.debug(
                    "headless error event provider=%s event=%s",
                    provider_name,
                    event.model_dump(mode="json", exclude_none=True),
                )
                raise HeadlessExecutionError(
                    f"{provider_name} command returned error event: {event.error_message()[:200]}"
                )

        text_chunks: list[str] = []
        usage: dict[str, int] = {}
        finish_reason: str | None = None
        for event in self.events:
            event_text = event.agent_text()
            if event_text is not None:
                text_chunks.append(event_text)
            event_usage = event.usage_payload()
            if event_usage is not None:
                usage = event_usage
                finish_reason = "stop"

        body: dict[str, Any] = {
            "text": "\n".join(text_chunks).strip(),
            "finish_reason": finish_reason,
            "usage": usage,
        }
        if self.passthrough_lines:
            body["non_json_stdout_lines"] = self.passthrough_lines
        raw_json = {
            "events": [
                event.model_dump(mode="json", exclude_none=True)
                for event in self.events
            ],
            "non_json_stdout_lines": self.passthrough_lines,
            "stderr": self.stderr,
        }
        return ParsedHeadlessOutput.from_body(body=body, raw_json=raw_json)


class CodexHeadlessAdapter(BaseHeadlessAdapter):
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

    def command_for_request(
        self,
        request: LlmRequest,
        payload: HeadlessRequestPayload,
        reasoning_mapping: ReasoningMappingResult,
    ) -> list[str]:
        del payload, reasoning_mapping
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

    def reasoning_mapping(self, request: LlmRequest) -> ReasoningMappingResult:
        return map_reasoning_for_codex_headless(
            request.reasoning,
            provider=request.provider,
            mode=CallMode.headless,
        )

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
        return CodexHeadlessResponse.from_stdout(
            stdout=stdout,
            stderr=stderr,
        ).to_parsed_output(provider_name=self.name)
