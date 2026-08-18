"""Regression coverage for the shared tool-schema invocation interface."""

from __future__ import annotations

import asyncio

import pytest

from box_agent.tools.base import (
    EventEmittingTool,
    Tool,
    ToolInvocationContext,
    ToolResult,
)


class RecordingTool(Tool):
    def __init__(self) -> None:
        self.calls: list[dict] = []

    @property
    def name(self) -> str:
        return "record"

    @property
    def description(self) -> str:
        return "Record one validated invocation."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "minLength": 1},
                "mode": {"type": "string", "enum": ["safe", "fast"]},
            },
            "required": ["text"],
            "additionalProperties": False,
        }

    async def execute(self, text: str, mode: str = "safe") -> ToolResult:
        self.calls.append({"text": text, "mode": mode})
        return ToolResult(success=True, content=f"{mode}:{text}")


class RecordingEventTool(EventEmittingTool):
    def __init__(self) -> None:
        super().__init__()
        self.context: tuple[asyncio.Queue, str, str] | None = None

    @property
    def name(self) -> str:
        return "event_record"

    @property
    def description(self) -> str:
        return "Record an event-aware invocation."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        }

    async def execute(self, text: str) -> ToolResult:
        raise AssertionError("event-aware invocation should use its context")

    async def execute_with_event_context(
        self,
        *,
        event_queue: asyncio.Queue,
        parent_tool_call_id: str,
        text: str,
    ) -> ToolResult:
        self.context = (event_queue, parent_tool_call_id, text)
        return ToolResult(success=True, content=text)


@pytest.mark.asyncio
async def test_invoke_validates_and_executes_valid_arguments_once() -> None:
    tool = RecordingTool()

    result = await tool.invoke({"text": "hello", "mode": "fast"})

    assert result.success is True
    assert result.content == "fast:hello"
    assert tool.calls == [{"text": "hello", "mode": "fast"}]


@pytest.mark.asyncio
async def test_invoke_rejects_invalid_arguments_without_executing_or_leaking_values() -> None:
    tool = RecordingTool()
    secret = "TOP_SECRET_VALUE"

    result = await tool.invoke(
        {"text": 123, "mode": "unsupported", "secret_field": secret}
    )

    assert result.success is False
    assert result.raw_output["code"] == "INVALID_TOOL_ARGUMENTS"
    assert {issue["keyword"] for issue in result.raw_output["issues"]} == {
        "additionalProperties",
        "enum",
        "type",
    }
    assert secret not in (result.error or "")
    assert secret not in str(result.raw_output)
    assert tool.calls == []


@pytest.mark.asyncio
async def test_invoke_rejects_invalid_schema_without_leaking_arguments() -> None:
    class InvalidSchemaTool(RecordingTool):
        @property
        def parameters(self) -> dict:
            return {"type": "definitely-not-a-json-schema-type"}

    tool = InvalidSchemaTool()
    secret = "TOP_SECRET_ARGUMENT_VALUE"

    result = await tool.invoke({"text": secret})

    assert result.success is False
    assert result.error == (
        "INVALID_TOOL_SCHEMA: record\n"
        "- /: tool parameter schema is invalid"
    )
    assert result.raw_output == {
        "code": "INVALID_TOOL_SCHEMA",
        "tool": "record",
        "issues": [
            {
                "path": "/",
                "keyword": "schema",
                "message": "tool parameter schema is invalid",
            }
        ],
    }
    assert secret not in (result.error or "")
    assert secret not in str(result.raw_output)
    assert tool.calls == []


@pytest.mark.asyncio
async def test_invoke_reports_missing_required_property_at_its_pointer() -> None:
    result = await RecordingTool().invoke({})

    assert result.success is False
    assert result.raw_output["issues"] == [
        {
            "path": "/text",
            "keyword": "required",
            "message": "required property 'text' is missing",
        }
    ]


@pytest.mark.asyncio
async def test_invoke_reports_each_missing_required_property_once() -> None:
    class TwoRequiredFieldsTool(RecordingTool):
        @property
        def parameters(self) -> dict:
            return {
                "type": "object",
                "properties": {
                    "first": {"type": "string"},
                    "second": {"type": "string"},
                },
                "required": ["first", "second"],
            }

    result = await TwoRequiredFieldsTool().invoke({})

    assert [issue["path"] for issue in result.raw_output["issues"]] == [
        "/first",
        "/second",
    ]


@pytest.mark.asyncio
async def test_event_tool_invocation_preserves_runtime_context() -> None:
    tool = RecordingEventTool()
    queue: asyncio.Queue = asyncio.Queue()

    result = await tool.invoke(
        {"text": "hello"},
        context=ToolInvocationContext(
            event_queue=queue,
            parent_tool_call_id="call-1",
        ),
    )

    assert result.success is True
    assert tool.context == (queue, "call-1", "hello")
