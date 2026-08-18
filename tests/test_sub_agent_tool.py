"""Test cases for SubAgentTool."""

from __future__ import annotations

import asyncio
from time import perf_counter
from unittest.mock import AsyncMock, MagicMock

import pytest
import tiktoken

from box_agent.events import DoneEvent, StopReason, SubAgentEvent, WebSearchEvent
from box_agent.context_resources import ResourceDescriptor
from box_agent.schema import LLMResponse, Message, StreamEvent, TokenUsage
from box_agent.agent import Agent
from box_agent.tools.base import Tool, ToolResult
from box_agent.tools.file_tools import ReadTool, WriteTool
from box_agent.tools.skill_loader import SkillLoader
from box_agent.tools.sub_agent_tool import SubAgentTool


# ── Helpers ──────────────────────────────────────────────────


class DummyTool(Tool):
    """A trivial tool for tests."""

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def description(self) -> str:
        return "A dummy tool"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, content="dummy result")


class WebSearchTool(Tool):
    """A web_search-shaped tool that returns reference metadata."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {"query": {"type": "string"}}}

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(
            success=True,
            content='{"refs":[{"reference_tag":"ref_1","title":"Example","url":"https://example.com"}]}',
        )


def _make_llm(text: str = "summary", tool_calls=None):
    """Return a mock LLM whose generate_stream yields the given text then finishes."""
    llm = AsyncMock()

    async def fake_stream(*, messages, tools, **kwargs):
        yield StreamEvent(type="text", delta=text)
        yield StreamEvent(
            type="finish",
            finish_reason="stop" if not tool_calls else "tool_use",
            tool_calls=tool_calls,
        )

    llm.generate_stream = fake_stream
    return llm


# ── Basic properties ─────────────────────────────────────────


def test_name():
    llm = AsyncMock()
    tool = SubAgentTool(llm=llm, parent_tools={})
    assert tool.name == "sub_agent"


def test_parallel_safe():
    llm = AsyncMock()
    tool = SubAgentTool(llm=llm, parent_tools={})
    assert tool.parallel_safe is True


def test_automatic_child_routing_selects_from_host_allowlist():
    class RoutingLLM:
        auto_model_candidates = (
            {
                "model": "model-general",
                "tags": ["general", "code"],
                "abilityLevel": 2,
            },
            {
                "model": "model-html",
                "tags": ["html", "frontend"],
                "abilityLevel": 2,
                "maxTokens": 50000,
            },
        )

        def __init__(self):
            self.bound = None

        def for_model(self, model, *, max_output_tokens=None):
            self.bound = (model, max_output_tokens)
            return f"bound:{model}"

    llm = RoutingLLM()
    tool = SubAgentTool(llm=llm, parent_tools={})

    child_llm, diagnostic = tool._resolve_task_llm(
        task="制作一个 HTML 前端页面",
        strategy="general_loop",
    )

    assert child_llm == "bound:model-html"
    assert llm.bound == ("model-html", 50000)
    assert diagnostic["selected_model"] == "model-html"
    assert diagnostic["task_tags"] == ["frontend", "html"]


def test_manual_child_routing_inherits_parent_model():
    class ManualLLM:
        pass

    llm = ManualLLM()
    tool = SubAgentTool(llm=llm, parent_tools={})

    child_llm, diagnostic = tool._resolve_task_llm(
        task="制作一个 HTML 前端页面",
        strategy="general_loop",
    )

    assert child_llm is llm
    assert diagnostic == {"mode": "inherit", "reason": "no_auto_model_pool"}


def test_default_parallel_safe_is_false():
    """Other tools should have parallel_safe == False by default."""
    dummy = DummyTool()
    assert dummy.parallel_safe is False


def test_schema():
    llm = AsyncMock()
    tool = SubAgentTool(llm=llm, parent_tools={})
    schema = tool.to_schema()
    assert schema["name"] == "sub_agent"
    assert "task" in schema["input_schema"]["properties"]
    # `title` is an optional short distinct label, not required.
    assert "title" in schema["input_schema"]["properties"]
    assert "capabilities" in schema["input_schema"]["properties"]
    assert "execution" in schema["input_schema"]["properties"]
    assert "batch_files" in schema["input_schema"]["properties"]["execution"]["properties"]["strategy"]["enum"]
    assert schema["input_schema"]["required"] == ["task"]

    openai_schema = tool.to_openai_schema()
    assert openai_schema["function"]["name"] == "sub_agent"


def test_description_prefers_cost_aware_batching_and_parent_merge():
    llm = AsyncMock()
    tool = SubAgentTool(llm=llm, parent_tools={})
    description = tool.description

    assert "independent context, parallel latency, or evidence isolation" in description
    assert "minimum tools and Skills" in description
    assert "batch_files" in description
    assert "required_tools=[\"read_file\"]" in description
    assert "fewest mutually exclusive batches" in description
    assert "Do not create multiple children merely because there are five or more units" in description
    assert "parent remains responsible" in description
    assert "final deliverables" in description
    assert "read_only:false" in description
    assert 'write_scope:["research/dim01.md"]' in description
    assert "Pass `budget` as an object" in description
    assert "never pass serialized JSON text" in description

    parameters = tool.parameters["properties"]
    assert "Never pass a serialized JSON string" in parameters["budget"]["description"]
    assert "Defaults are read_only=true" in parameters["constraints"]["description"]
    assert "mutually exclusive paths" in parameters["constraints"]["properties"]["write_scope"]["description"]


# ── Tool filtering ───────────────────────────────────────────


def test_child_tools_exclude_sub_agent():
    """SubAgentTool must not include itself in the child tool set."""
    llm = AsyncMock()
    dummy = DummyTool()
    parent = {"dummy": dummy, "sub_agent": SubAgentTool(llm=llm, parent_tools={})}
    tool = SubAgentTool(llm=llm, parent_tools=parent)
    resolved = tool._resolve_child_tools()
    assert "sub_agent" not in resolved
    assert "dummy" in resolved


def test_resolve_child_tools_prefers_live_provider():
    """Child toolset follows the parent's live tool map, not the snapshot.

    Tools registered after construction (e.g. MCP web_search merged in via
    register_mcp_tools) must be inherited by child agents.
    """
    llm = AsyncMock()
    snapshot = {"dummy": DummyTool()}
    tool = SubAgentTool(llm=llm, parent_tools=snapshot)

    # Live parent map gains a tool after construction; provider points at it.
    live: dict = {"dummy": DummyTool(), "sub_agent": tool}
    tool.set_tool_provider(lambda: live)
    live["web_search"] = DummyTool()  # simulate late MCP merge (in-place mutation)

    resolved = tool._resolve_child_tools()
    assert "web_search" in resolved  # late tool inherited
    assert "sub_agent" not in resolved  # still excludes itself


def test_resolve_child_tools_falls_back_to_snapshot():
    """Without a provider (or if it fails), fall back to the snapshot."""
    llm = AsyncMock()
    tool = SubAgentTool(llm=llm, parent_tools={"dummy": DummyTool()})
    assert "dummy" in tool._resolve_child_tools()  # no provider → snapshot

    tool.set_tool_provider(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert "dummy" in tool._resolve_child_tools()  # provider raised → snapshot


# ── Execution ────────────────────────────────────────────────


async def test_basic_execution():
    """Sub-agent returns the LLM's final text as ToolResult content."""
    llm = _make_llm(text="Analysis complete: revenue up 20%")
    tool = SubAgentTool(llm=llm, parent_tools={})
    result = await tool.execute(task="Analyze revenue data")
    assert result.success is True
    assert "revenue up 20%" in result.content


async def test_forwarded_events_carry_short_title():
    """A provided `title` becomes the SubAgentEvent label; task is unchanged."""
    llm = _make_llm(text="done")
    tool = SubAgentTool(llm=llm, parent_tools={})
    queue = asyncio.Queue()
    tool._event_queue = queue
    tool._parent_tool_call_id = "parent-sub-agent"

    await tool.execute(
        task="围绕商汤科技（SenseTime, 0020.HK）做一个独立研究切片：财务表现与业务结构",
        title="财务表现与业务结构",
    )

    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    sub_events = [e for e in events if isinstance(e, SubAgentEvent)]
    assert sub_events
    assert all(e.title == "财务表现与业务结构" for e in sub_events)
    # task_preview still reflects the (long, shared-prefix) task.
    assert all(e.title != e.task_preview for e in sub_events)


async def test_title_falls_back_to_task_preview_when_omitted():
    """Without a title, the label falls back to the task preview (no break)."""
    llm = _make_llm(text="done")
    tool = SubAgentTool(llm=llm, parent_tools={})
    queue = asyncio.Queue()
    tool._event_queue = queue
    tool._parent_tool_call_id = "parent-sub-agent"

    await tool.execute(task="Analyze revenue data for Q3")

    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    sub_events = [e for e in events if isinstance(e, SubAgentEvent)]
    assert sub_events
    assert all(e.title == e.task_preview for e in sub_events)


async def test_sub_agent_inherits_parent_system_prompt_constraints():
    """Child system prompt includes finalized parent instructions automatically."""
    captured_messages = None

    async def fake_stream(*, messages, tools, **kwargs):
        nonlocal captured_messages
        captured_messages = messages
        yield StreamEvent(type="text", delta="done")
        yield StreamEvent(type="finish", finish_reason="stop", tool_calls=None)

    llm = AsyncMock()
    llm.generate_stream = fake_stream

    parent_prompt = "Parent constraint: write drafts under draft-a/ only."
    tool = SubAgentTool(llm=llm, parent_tools={})
    tool.set_parent_system_prompt(parent_prompt)

    result = await tool.execute(task="Draft one isolated section")

    assert result.success is True
    assert captured_messages is not None
    child_system_prompt = captured_messages[0].content
    assert "Inherited parent system prompt" in child_system_prompt
    assert parent_prompt in child_system_prompt
    assert "Do not overwrite shared files or final deliverables" in child_system_prompt


def test_agent_wires_system_prompt_into_sub_agent(tmp_path):
    """Agent initialization attaches its finalized system prompt to SubAgentTool."""
    llm = AsyncMock()
    tool = SubAgentTool(llm=llm, parent_tools={})

    Agent(
        llm_client=llm,
        system_prompt="Parent constraint: keep output generic.",
        tools=[tool],
        workspace_dir=str(tmp_path),
    )

    assert tool._parent_system_prompt is not None
    assert "Parent constraint: keep output generic." in tool._parent_system_prompt
    assert "Current Workspace" in tool._parent_system_prompt


def test_sub_agent_prompt_replaces_parent_only_mcp_search_guidance(tmp_path):
    llm = AsyncMock()
    tool = SubAgentTool(llm=llm, parent_tools={})

    agent = Agent(
        llm_client=llm,
        system_prompt="Parent constraint.",
        tools=[tool],
        workspace_dir=str(tmp_path),
        deferred_mcp_loading_enabled=True,
    )

    assert tool._parent_system_prompt is not None
    assert "Use `tool_search`" not in tool._parent_system_prompt
    assert "The parent agent owns deferred MCP discovery" in tool._parent_system_prompt
    assert "tool_search" not in agent._inherited_tools()


async def test_sub_agent_read_ledger_is_local_to_child_context(tmp_path):
    from box_agent.schema import FunctionCall, ToolCall

    path = tmp_path / "reference.md"
    path.write_text("CHILD_EXACT_BODY\n", encoding="utf-8")
    captured_requests: list[list[Message]] = []
    call_count = 0

    async def fake_stream(*, messages, tools, **kwargs):
        nonlocal call_count
        call_count += 1
        captured_requests.append([message.model_copy(deep=True) for message in messages])
        if call_count == 1:
            yield StreamEvent(
                type="finish",
                finish_reason="tool_use",
                tool_calls=[
                    ToolCall(
                        id="child-read",
                        type="function",
                        function=FunctionCall(
                            name="read_file",
                            arguments={"path": "reference.md"},
                        ),
                    )
                ],
            )
            return
        yield StreamEvent(type="text", delta="done")
        yield StreamEvent(type="finish", finish_reason="stop")

    llm = AsyncMock()
    llm.generate_stream = fake_stream
    read_tool = ReadTool(workspace_dir=str(tmp_path))
    sub_agent = SubAgentTool(
        llm=llm,
        parent_tools={"read_file": read_tool},
        workspace_dir=str(tmp_path),
    )
    parent = Agent(
        llm_client=llm,
        system_prompt="parent",
        tools=[read_tool, sub_agent],
        workspace_dir=str(tmp_path),
    )
    parent_read = await read_tool.execute(path="reference.md")
    descriptor = ResourceDescriptor.from_raw_output(parent_read.raw_output)
    assert descriptor is not None
    parent.context_resource_ledger.register_full_source(
        "parent-read",
        descriptor,
        parent_read.content,
    )

    result = await sub_agent.execute(
        task="Read the reference",
        capabilities={"required_tools": ["read_file"]},
    )

    assert result.success is True
    child_tool_message = next(
        message for message in captured_requests[1] if message.role == "tool"
    )
    assert "CHILD_EXACT_BODY" in child_tool_message.content
    assert "Resource already available" not in child_tool_message.content
    assert parent.context_resource_ledger.source_ids == ("parent-read",)


async def test_new_style_general_loop_uses_only_resolved_tools_and_slim_prompt(tmp_path):
    captured = {}

    async def fake_stream(*, messages, tools, **kwargs):
        captured["messages"] = messages
        captured["tools"] = tools
        yield StreamEvent(type="text", delta="done")
        yield StreamEvent(
            type="finish",
            finish_reason="stop",
            tool_calls=None,
            usage=TokenUsage(prompt_tokens=11, completion_tokens=2, total_tokens=13),
        )

    llm = AsyncMock()
    llm.generate_stream = fake_stream
    read_file = ReadTool(workspace_dir=str(tmp_path))
    tool = SubAgentTool(
        llm=llm,
        parent_tools={"read_file": read_file, "web_search": WebSearchTool()},
        workspace_dir=str(tmp_path),
    )
    tool.set_parent_system_prompt("SECRET_PARENT_PROMPT with the full capability catalog")

    result = await tool.execute(
        task="Inspect the local inputs",
        capabilities={"required_tools": ["read_file"]},
    )

    assert result.success is True
    assert [candidate.name for candidate in captured["tools"]] == ["read_file"]
    system_prompt = captured["messages"][0].content
    assert "Immutable rules" in system_prompt
    assert "SECRET_PARENT_PROMPT" not in system_prompt
    assert "Inherited parent system prompt" not in system_prompt
    assert result.raw_output["legacy_general"] is False
    assert result.raw_output["resolved_tools"] == ["read_file"]
    assert result.raw_output["model_calls"] == 1
    assert result.raw_output["usage"] == {
        "input_tokens": 11,
        "output_tokens": 2,
        "total_tokens": 13,
    }


async def test_invalid_new_style_spec_never_falls_back_or_calls_llm():
    llm = AsyncMock()
    tool = SubAgentTool(llm=llm, parent_tools={})

    result = await tool.execute(task="Inspect", capabilities={})

    assert result.success is False
    assert result.raw_output["code"] == "INVALID_DELEGATION_SPEC"
    assert result.raw_output["retryable"] is True
    assert "minimal_valid_example" in result.raw_output
    assert result.raw_output["retry_limit"] == 1
    llm.generate.assert_not_called()
    llm.generate_stream.assert_not_called()


async def test_invalid_budget_string_returns_object_correction_example():
    llm = AsyncMock()
    tool = SubAgentTool(llm=llm, parent_tools={})

    result = await tool.execute(
        task="Research one dimension",
        capabilities={"required_tools": ["read_file"]},
        budget='{"max_steps": 12, "max_tool_calls": 25}',
    )

    assert result.success is False
    assert result.raw_output["code"] == "INVALID_DELEGATION_SPEC"
    assert result.raw_output["invalid_fields"] == ["budget"]
    assert result.raw_output["field_corrections"]["budget"] == {
        "message": "Pass budget as a JSON object, never as a JSON string.",
        "example": {"max_steps": 12, "max_tool_calls": 25},
    }
    llm.generate_stream.assert_not_called()


async def test_write_conflict_returns_scoped_write_correction_hint(tmp_path):
    llm = AsyncMock()
    tool = SubAgentTool(
        llm=llm,
        parent_tools={"write_file": WriteTool(workspace_dir=str(tmp_path))},
        workspace_dir=str(tmp_path),
    )

    result = await tool.execute(
        task="Write one research dimension",
        capabilities={"required_tools": ["write_file"]},
    )

    assert result.success is False
    assert result.raw_output["code"] == "CAPABILITY_CONSTRAINT_CONFLICT"
    assert "constraints.read_only=false" in result.raw_output["correction_hint"]
    assert "constraints.write_scope" in result.raw_output["correction_hint"]
    assert "external_side_effect=false" in result.raw_output["correction_hint"]
    llm.generate_stream.assert_not_called()


async def test_explicit_null_capabilities_is_invalid_not_legacy():
    llm = AsyncMock()
    tool = SubAgentTool(llm=llm, parent_tools={})

    result = await tool.execute(task="Inspect", capabilities=None)

    assert result.success is False
    assert result.raw_output["code"] == "INVALID_DELEGATION_SPEC"
    assert "capabilities" in result.raw_output["invalid_fields"]
    llm.generate_stream.assert_not_called()


def test_sub_agent_schema_rejects_unknown_top_level_fields():
    tool = SubAgentTool(llm=AsyncMock(), parent_tools={})

    assert tool.parameters["additionalProperties"] is False


async def test_unknown_top_level_fields_return_structured_failure():
    llm = AsyncMock()
    tool = SubAgentTool(llm=llm, parent_tools={})

    result = await tool.execute(
        task="Inspect",
        required_tools='["file", "terminal"]',
        capabilities="",
    )

    assert result.success is False
    assert result.raw_output["code"] == "INVALID_DELEGATION_SPEC"
    assert result.raw_output["invalid_fields"] == ["capabilities", "required_tools"]
    assert "Traceback" not in (result.error or "")
    llm.generate.assert_not_called()
    llm.generate_stream.assert_not_called()


async def test_event_context_missing_task_returns_structured_failure():
    llm = AsyncMock()
    tool = SubAgentTool(llm=llm, parent_tools={})

    result = await tool.execute_with_event_context(
        event_queue=asyncio.Queue(),
        parent_tool_call_id="parent-call",
        capabilities={"required_tools": ["read_file"]},
    )

    assert result.success is False
    assert result.raw_output["code"] == "INVALID_DELEGATION_SPEC"
    assert result.raw_output["invalid_fields"] == ["task"]
    assert "Traceback" not in (result.error or "")
    llm.generate.assert_not_called()
    llm.generate_stream.assert_not_called()


async def test_selected_skills_are_loaded_into_new_prompt_only(tmp_path):
    skill_dir = tmp_path / "review-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        """---
name: review-skill
description: Review local material
allowed-tools: [read_file]
---

Follow the REVIEW-SKILL-CONTENT rubric.
""",
        encoding="utf-8",
    )
    loader = SkillLoader(tmp_path)
    loader.discover_skills()
    captured_messages = None

    async def fake_stream(*, messages, tools, **kwargs):
        nonlocal captured_messages
        captured_messages = messages
        yield StreamEvent(type="text", delta="done")
        yield StreamEvent(type="finish", finish_reason="stop", tool_calls=None)

    llm = AsyncMock()
    llm.generate_stream = fake_stream
    tool = SubAgentTool(
        llm=llm,
        parent_tools={"read_file": ReadTool(workspace_dir=str(tmp_path))},
        workspace_dir=str(tmp_path),
    )
    tool.set_skill_provider(lambda: loader)

    result = await tool.execute(
        task="Review the material",
        capabilities={
            "required_tools": ["read_file"],
            "skills": ["review-skill"],
        },
    )

    assert result.success is True
    assert "REVIEW-SKILL-CONTENT" in captured_messages[0].content
    assert result.raw_output["resolved_skills"] == ["review-skill"]


async def test_capability_state_provider_drives_not_ready_error():
    llm = AsyncMock()
    tool = SubAgentTool(llm=llm, parent_tools={})
    tool.set_capability_state_provider(lambda: "loading")

    result = await tool.execute(
        task="Use the future MCP tool",
        capabilities={"required_tools": ["mcp_future_tool"]},
        constraints={
            "read_only": False,
            "network": True,
            "write_scope": None,
            "external_side_effect": True,
        },
    )

    assert result.success is False
    assert result.raw_output["code"] == "REQUIRED_TOOL_NOT_READY"
    assert result.raw_output["pending_source"] == "mcp"
    assert result.raw_output["requested_tools"]["required"] == ["mcp_future_tool"]
    assert result.raw_output["resolved_tools"] == []
    assert result.raw_output["denied_tools"][0]["name"] == "mcp_future_tool"
    assert result.raw_output["model_calls"] == 0
    llm.generate_stream.assert_not_called()


async def test_web_search_tool_emits_reference_event():
    """web_search tool results should surface refs as a structured event."""
    from box_agent.core import run_agent_loop
    from box_agent.schema import FunctionCall, ToolCall

    call_num = 0

    async def fake_stream(*, messages, tools, **kwargs):
        nonlocal call_num
        call_num += 1
        if call_num == 1:
            yield StreamEvent(
                type="finish",
                finish_reason="tool_use",
                tool_calls=[
                    ToolCall(
                        id="search-1",
                        type="function",
                        function=FunctionCall(name="web_search", arguments={"query": "example"}),
                    )
                ],
            )
        else:
            yield StreamEvent(type="text", delta="summary [ref_1]")
            yield StreamEvent(type="finish", finish_reason="stop", tool_calls=None)

    llm = AsyncMock()
    llm.generate_stream = fake_stream

    events = []
    async for event in run_agent_loop(
        llm=llm,
        messages=[Message(role="user", content="search")],
        tools={"web_search": WebSearchTool()},
        max_steps=3,
    ):
        events.append(event)

    web_events = [event for event in events if isinstance(event, WebSearchEvent)]
    assert len(web_events) == 1
    assert web_events[0].tool_call_id == "search-1"
    assert web_events[0].payload["refs"][0]["reference_tag"] == "ref_1"


async def test_sub_agent_forwards_web_search_reference_event():
    """Sub-agent child web_search refs should be forwarded to the parent stream."""
    from box_agent.schema import FunctionCall, ToolCall

    call_num = 0

    async def fake_stream(*, messages, tools, **kwargs):
        nonlocal call_num
        call_num += 1
        if call_num == 1:
            yield StreamEvent(
                type="finish",
                finish_reason="tool_use",
                tool_calls=[
                    ToolCall(
                        id="child-search-1",
                        type="function",
                        function=FunctionCall(name="web_search", arguments={"query": "example"}),
                    )
                ],
            )
        else:
            yield StreamEvent(type="text", delta="child summary [ref_1]")
            yield StreamEvent(type="finish", finish_reason="stop", tool_calls=None)

    llm = AsyncMock()
    llm.generate_stream = fake_stream

    tool = SubAgentTool(llm=llm, parent_tools={"web_search": WebSearchTool()})

    queue = asyncio.Queue()
    tool._event_queue = queue
    tool._parent_tool_call_id = "parent-sub-agent"

    result = await tool.execute(task="search in child")

    forwarded = []
    while not queue.empty():
        forwarded.append(queue.get_nowait())

    assert result.success is True
    web_events = [
        event
        for event in forwarded
        if isinstance(event, SubAgentEvent) and isinstance(event.event, WebSearchEvent)
    ]
    assert len(web_events) == 1
    assert web_events[0].parent_tool_call_id == "parent-sub-agent"
    assert web_events[0].sub_agent_id.startswith("subagent-")
    assert web_events[0].event.payload["refs"][0]["url"] == "https://example.com"


async def test_empty_output_returns_error():
    """If the LLM produces no content, the tool should report failure."""
    llm = _make_llm(text="")
    tool = SubAgentTool(llm=llm, parent_tools={})
    result = await tool.execute(task="Do something")
    assert result.success is False
    assert "without producing output" in result.error


async def test_llm_exception_returns_error():
    """If the LLM raises, ToolResult should contain the error info."""
    llm = AsyncMock()

    async def boom(*, messages, tools, **kwargs):
        raise RuntimeError("API timeout")
        yield  # make it an async generator  # noqa: E501

    llm.generate_stream = boom
    tool = SubAgentTool(llm=llm, parent_tools={})
    result = await tool.execute(task="Try this")
    # run_agent_loop catches the exception and yields DoneEvent with error as final_content,
    # so SubAgentTool wraps it as a successful result containing the error text.
    # The error is humanized via classify_llm_error: "API timeout" classifies as a
    # timeout, so the friendly Chinese message is surfaced rather than the raw string.
    assert "超时" in result.content


async def test_max_steps_respected():
    """Sub-agent should stop after max_steps even if LLM keeps requesting tools."""
    from box_agent.schema import FunctionCall, ToolCall

    call_count = 0

    async def looping_stream(*, messages, tools, **kwargs):
        nonlocal call_count
        call_count += 1
        # Always request a tool call to keep the loop going
        tc = ToolCall(
            id=f"tc-{call_count}",
            type="function",
            function=FunctionCall(name="dummy", arguments={}),
        )
        yield StreamEvent(type="text", delta=f"step {call_count}")
        yield StreamEvent(type="finish", finish_reason="tool_use", tool_calls=[tc])

    llm = AsyncMock()
    llm.generate_stream = looping_stream

    dummy = DummyTool()
    tool = SubAgentTool(
        llm=llm,
        parent_tools={"dummy": dummy},
        max_steps=3,
    )
    result = await tool.execute(task="Loop forever")
    # Should have stopped — call_count should be capped at max_steps
    assert call_count <= 4  # max_steps=3 means 3 LLM calls


async def test_batch_files_reads_twenty_files_once_and_calls_generate_once(tmp_path):
    paths = []
    for index in range(20):
        path = tmp_path / f"project-{index:02d}.md"
        path.write_text(f"# Project {index}\nScore material {index}\n", encoding="utf-8")
        paths.append(path.name)

    class CountingReadTool(ReadTool):
        def __init__(self):
            super().__init__(workspace_dir=str(tmp_path))
            self.calls = []

        async def execute(self, path, offset=None, limit=None):
            self.calls.append(path)
            return await super().execute(path=path, offset=offset, limit=limit)

    class BatchLLM:
        def __init__(self):
            self.generate_calls = 0
            self.stream_calls = 0
            self.messages = None
            self.tools = "unset"
            self.generate_kwargs = None

        async def generate(self, messages, tools=None, **kwargs):
            self.generate_calls += 1
            self.messages = messages
            self.tools = tools
            self.generate_kwargs = kwargs
            encoding = tiktoken.get_encoding("cl100k_base")
            prompt_tokens = sum(
                len(encoding.encode(str(message.content))) for message in messages
            )
            return LLMResponse(
                content="ranked all 20 projects",
                finish_reason="stop",
                usage=TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=100,
                    total_tokens=prompt_tokens + 100,
                ),
            )

        async def generate_stream(self, messages, tools=None, **kwargs):
            self.stream_calls += 1
            raise AssertionError("batch_files must not enter run_agent_loop")
            yield

    llm = BatchLLM()
    read_tool = CountingReadTool()
    tool = SubAgentTool(
        llm=llm,
        parent_tools={"read_file": read_tool},
        workspace_dir=str(tmp_path),
    )

    started = perf_counter()
    result = await tool.execute(
        task="Compare every project and rank them",
        execution={"strategy": "batch_files"},
        capabilities={"required_tools": ["read_file"]},
        inputs={"files": list(reversed(paths))},
    )
    elapsed = perf_counter() - started

    assert result.success is True
    assert result.content == "ranked all 20 projects"
    assert sorted(read_tool.calls) == sorted(paths)
    assert len(read_tool.calls) == 20
    assert llm.generate_calls == 1
    assert llm.stream_calls == 0
    assert llm.tools is None
    assert "session_id" not in llm.generate_kwargs
    assert llm.generate_kwargs["call_kind"] == "subagent_step"
    assert "<<<UNTRUSTED_FILE" in llm.messages[-1].content
    assert llm.messages[-1].content.count("<<<UNTRUSTED_FILE") == 20
    assert result.raw_output["model_calls"] == 1
    assert result.raw_output["tool_calls"] == 20
    assert result.raw_output["resolved_tools"] == ["read_file"]
    assert result.raw_output["usage"]["input_tokens"] <= int(3_353_714 * 0.10)
    assert elapsed < 60


async def test_batch_files_prefetch_propagates_cancellation():
    started = asyncio.Event()

    class CancellableReadTool(Tool):
        @property
        def name(self):
            return "read_file"

        @property
        def description(self):
            return "test read"

        @property
        def parameters(self):
            return {"type": "object", "properties": {"path": {"type": "string"}}}

        async def execute(self, path):
            started.set()
            await asyncio.Event().wait()

    llm = AsyncMock()
    tool = SubAgentTool(
        llm=llm,
        parent_tools={"read_file": CancellableReadTool()},
    )

    execution = asyncio.create_task(
        tool.execute(
            task="Summarize",
            execution={"strategy": "batch_files"},
            capabilities={"required_tools": ["read_file"]},
            inputs={"files": ["one.md"]},
        )
    )
    await started.wait()
    execution.cancel()

    with pytest.raises(asyncio.CancelledError):
        await execution
    llm.generate.assert_not_called()


async def test_batch_files_uses_configurable_synthesis_timeout():
    class CompleteReadTool(Tool):
        @property
        def name(self):
            return "read_file"

        @property
        def description(self):
            return "test read"

        @property
        def parameters(self):
            return {"type": "object", "properties": {"path": {"type": "string"}}}

        async def execute(self, path):
            return ToolResult(
                success=True,
                content="body",
                raw_output={
                    "source_char_count": 4,
                    "selected_char_count": 4,
                    "selected_line_count": 1,
                    "truncated": False,
                },
            )

    class HangingLLM:
        async def generate(self, **kwargs):
            await asyncio.Event().wait()

    tool = SubAgentTool(
        llm=HangingLLM(),
        parent_tools={"read_file": CompleteReadTool()},
        batch_synthesis_timeout_seconds=0.01,
    )

    result = await tool.execute(
        task="Summarize",
        execution={"strategy": "batch_files"},
        capabilities={"required_tools": ["read_file"]},
        inputs={"files": ["one.md"]},
    )

    assert result.success is False
    assert result.raw_output["code"] == "BATCH_SYNTHESIS_TIMEOUT"
    assert result.raw_output["timeout_seconds"] == 0.01
    assert "configured 0.01 second runtime limit" in result.raw_output["message"]


@pytest.mark.parametrize(
    ("content", "raw_output", "expected_code"),
    [
        (
            "normal body",
            None,
            "READ_COMPLETENESS_UNVERIFIED",
        ),
        (
            "body ... [Content truncated: 40000 tokens -> ~32000 tokens limit] ...",
            None,
            "FILE_CONTENT_TRUNCATED",
        ),
        (
            "large body",
            {
                "source_char_count": 64_001,
                "selected_char_count": 64_001,
                "selected_line_count": 1,
                "truncated": False,
            },
            "FILE_TOO_LARGE",
        ),
        (
            "truncated body",
            {
                "source_char_count": 10,
                "selected_char_count": 10,
                "selected_line_count": 1,
                "truncated": True,
            },
            "FILE_CONTENT_TRUNCATED",
        ),
    ],
)
async def test_batch_files_rejects_unproven_or_incomplete_reads_before_model(
    content,
    raw_output,
    expected_code,
):
    class UnsafeReadTool(Tool):
        @property
        def name(self):
            return "read_file"

        @property
        def description(self):
            return "test read"

        @property
        def parameters(self):
            return {"type": "object", "properties": {"path": {"type": "string"}}}

        async def execute(self, path):
            return ToolResult(success=True, content=content, raw_output=raw_output)

    llm = AsyncMock()
    tool = SubAgentTool(llm=llm, parent_tools={"read_file": UnsafeReadTool()})

    result = await tool.execute(
        task="Summarize",
        execution={"strategy": "batch_files"},
        capabilities={"required_tools": ["read_file"]},
        inputs={"files": ["one.md"]},
    )

    assert result.success is False
    assert result.raw_output["type"] == "sub_agent_delegation_error"
    assert result.raw_output["code"] == "BATCH_FILES_PREFETCH_FAILED"
    assert result.raw_output["failures"][0]["code"] == expected_code
    assert result.raw_output["model_calls"] == 0
    llm.generate.assert_not_called()
    llm.generate_stream.assert_not_called()


async def test_batch_files_rejects_aggregate_over_200k_before_model():
    class LargeCompleteReadTool(Tool):
        @property
        def name(self):
            return "read_file"

        @property
        def description(self):
            return "test read"

        @property
        def parameters(self):
            return {"type": "object", "properties": {"path": {"type": "string"}}}

        async def execute(self, path):
            content = "x" * 51_000
            return ToolResult(
                success=True,
                content=content,
                raw_output={
                    "source_char_count": len(content),
                    "selected_char_count": len(content),
                    "selected_line_count": 1,
                    "truncated": False,
                },
            )

    llm = AsyncMock()
    tool = SubAgentTool(
        llm=llm,
        parent_tools={"read_file": LargeCompleteReadTool()},
    )

    result = await tool.execute(
        task="Summarize all files",
        execution={"strategy": "batch_files"},
        capabilities={"required_tools": ["read_file"]},
        inputs={"files": ["a.md", "b.md", "c.md", "d.md"]},
    )

    assert result.success is False
    assert result.raw_output["failures"] == [
        {
            "path": "*",
            "code": "AGGREGATE_CONTENT_TOO_LARGE",
            "source_char_count": 204_000,
            "limit": 200_000,
            "retryable": False,
        }
    ]
    assert result.raw_output["model_calls"] == 0
    llm.generate.assert_not_called()


async def test_write_scope_is_enforced_before_live_write_tool(tmp_path):
    from box_agent.schema import FunctionCall, ToolCall

    call_num = 0

    async def fake_stream(*, messages, tools, **kwargs):
        nonlocal call_num
        call_num += 1
        if call_num == 1:
            yield StreamEvent(
                type="finish",
                finish_reason="tool_use",
                tool_calls=[
                    ToolCall(
                        id="write-1",
                        type="function",
                        function=FunctionCall(
                            name="write_file",
                            arguments={"path": "../outside.txt", "content": "blocked"},
                        ),
                    )
                ],
            )
        else:
            yield StreamEvent(type="text", delta="write was denied")
            yield StreamEvent(type="finish", finish_reason="stop", tool_calls=None)

    llm = AsyncMock()
    llm.generate_stream = fake_stream
    write_tool = WriteTool(workspace_dir=str(tmp_path))
    tool = SubAgentTool(
        llm=llm,
        parent_tools={"write_file": write_tool},
        workspace_dir=str(tmp_path),
    )

    result = await tool.execute(
        task="Write only inside allowed",
        capabilities={"required_tools": ["write_file"]},
        constraints={
            "read_only": False,
            "network": False,
            "write_scope": ["allowed"],
            "external_side_effect": False,
        },
    )

    assert result.success is True
    assert not (tmp_path.parent / "outside.txt").exists()
    assert result.raw_output["tool_calls"] == 1


# ── Parallel execution in core ───────────────────────────────


async def test_parallel_execution_in_core():
    """Multiple parallel_safe tool calls should be gathered concurrently."""
    import asyncio
    from box_agent.core import run_agent_loop
    from box_agent.schema import FunctionCall, ToolCall

    execution_order = []

    class SlowSubAgent(Tool):
        parallel_safe = True

        @property
        def name(self) -> str:
            return "sub_agent"

        @property
        def description(self) -> str:
            return "test"

        @property
        def parameters(self) -> dict:
            return {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]}

        async def execute(self, task: str) -> ToolResult:
            execution_order.append(f"start:{task}")
            await asyncio.sleep(0.05)
            execution_order.append(f"end:{task}")
            return ToolResult(success=True, content=f"Done: {task}")

    # LLM: first call returns 2 sub_agent tool calls, second call ends
    call_num = 0

    async def fake_stream(*, messages, tools, **kwargs):
        nonlocal call_num
        call_num += 1
        if call_num == 1:
            yield StreamEvent(type="text", delta="Delegating")
            yield StreamEvent(
                type="finish",
                finish_reason="tool_use",
                tool_calls=[
                    ToolCall(id="tc-1", type="function", function=FunctionCall(name="sub_agent", arguments={"task": "A"})),
                    ToolCall(id="tc-2", type="function", function=FunctionCall(name="sub_agent", arguments={"task": "B"})),
                ],
            )
        else:
            yield StreamEvent(type="text", delta="All done")
            yield StreamEvent(type="finish", finish_reason="stop", tool_calls=None)

    llm = AsyncMock()
    llm.generate_stream = fake_stream

    messages = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Do two things"),
    ]
    tools = {"sub_agent": SlowSubAgent()}

    events = []
    async for event in run_agent_loop(llm=llm, messages=messages, tools=tools, max_steps=5):
        events.append(event)

    # Both starts should appear before either result (parallel execution)
    start_events = [e for e in events if hasattr(e, "tool_name") and hasattr(e, "arguments") and not hasattr(e, "success")]
    result_events = [e for e in events if hasattr(e, "success") and hasattr(e, "tool_name")]

    sub_starts = [e for e in start_events if e.tool_name == "sub_agent"]
    sub_results = [e for e in result_events if e.tool_name == "sub_agent"]

    assert len(sub_starts) == 2
    assert len(sub_results) == 2

    # Verify parallel execution: both starts happen before both ends
    assert execution_order[0].startswith("start:")
    assert execution_order[1].startswith("start:")


@pytest.mark.asyncio
async def test_parallel_sub_agent_progress_keeps_parent_tool_call_id():
    class TaskEchoLLM:
        async def generate(self, messages, tools=None):
            return LLMResponse(content=f"done {messages[-1].content}", finish_reason="stop")

        async def generate_stream(self, messages, tools=None, **kwargs):
            task = messages[-1].content
            await asyncio.sleep(0.02 if task == "A" else 0.01)
            yield StreamEvent(type="text", delta=f"done {task}")
            yield StreamEvent(type="finish", finish_reason="stop", tool_calls=None)

    tool = SubAgentTool(llm=TaskEchoLLM(), parent_tools={}, max_steps=1)
    queue: asyncio.Queue[SubAgentEvent] = asyncio.Queue()

    result_a, result_b = await asyncio.gather(
        tool.execute_with_event_context(
            event_queue=queue,
            parent_tool_call_id="parent-a",
            task="A",
            title="A",
        ),
        tool.execute_with_event_context(
            event_queue=queue,
            parent_tool_call_id="parent-b",
            task="B",
            title="B",
        ),
    )

    assert result_a.success is True
    assert result_b.success is True

    events: list[SubAgentEvent] = []
    while not queue.empty():
        events.append(queue.get_nowait())

    assert events
    assert {event.title for event in events} == {"A", "B"}
    assert {
        event.parent_tool_call_id
        for event in events
        if event.title == "A"
    } == {"parent-a"}
    assert {
        event.parent_tool_call_id
        for event in events
        if event.title == "B"
    } == {"parent-b"}


async def test_parallel_new_style_calls_do_not_leak_resolved_tools():
    observed = {}

    class IsolatedLLM:
        async def generate_stream(self, messages, tools=None, **kwargs):
            task_text = messages[-1].content
            key = "read" if "read task" in task_text else "web"
            observed[key] = [tool.name for tool in tools]
            await asyncio.sleep(0.01)
            yield StreamEvent(type="text", delta=f"done {key}")
            yield StreamEvent(type="finish", finish_reason="stop", tool_calls=None)

    tool = SubAgentTool(
        llm=IsolatedLLM(),
        parent_tools={
            "read_file": ReadTool(),
            "web_search": WebSearchTool(),
        },
    )
    result_read, result_web = await asyncio.gather(
        tool.execute(
            task="read task",
            capabilities={"required_tools": ["read_file"]},
        ),
        tool.execute(
            task="web task",
            capabilities={"required_tools": ["web_search"]},
            constraints={
                "read_only": True,
                "network": True,
                "write_scope": None,
                "external_side_effect": False,
            },
        ),
    )

    assert result_read.success is True
    assert result_web.success is True
    assert observed == {"read": ["read_file"], "web": ["web_search"]}
    assert result_read.raw_output["resolved_tools"] == ["read_file"]
    assert result_web.raw_output["resolved_tools"] == ["web_search"]


def test_add_workspace_tools_wires_sub_agent_token_limit(tmp_path) -> None:
    """Sub-agent config and live capability providers flow through setup."""
    from box_agent.config import AgentConfig, ToolLimitsConfig, ToolsConfig
    from box_agent.tools.setup import add_workspace_tools

    class Config:
        tool_limits = ToolLimitsConfig(
            sub_agent={"legacy_max_steps": 55, "no_progress_steps": 9}
        )
        agent = AgentConfig(
            sub_agent_token_limit=12345,
            sub_agent_batch_synthesis_timeout_seconds=234.5,
        )
        tools = ToolsConfig(
            enable_bash=False,
            enable_file_tools=False,
            enable_todo=False,
            enable_sub_agent=True,
        )

    tools: list = []
    skill_loader = object()
    add_workspace_tools(
        tools,
        Config(),
        tmp_path,
        allow_full_access=False,
        llm=AsyncMock(),
        skill_loader=skill_loader,
        capability_state_provider=lambda: "loading",
        output=lambda *_: None,
    )

    sub_agent = next(t for t in tools if t.name == "sub_agent")
    assert sub_agent._token_limit == 12345
    assert sub_agent._max_steps == 55
    assert sub_agent._no_progress_limit == 9
    assert sub_agent._batch_synthesis_timeout_seconds == 234.5
    assert sub_agent._resolve_skill_loader() is skill_loader
    assert sub_agent._resolve_capability_state() == "loading"
