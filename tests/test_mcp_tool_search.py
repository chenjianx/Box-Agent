"""Focused tests for deferred MCP discovery and per-session exposure."""

from __future__ import annotations

import asyncio
import json
from collections import OrderedDict

import pytest

from box_agent.agent import Agent
from box_agent.events import ToolCallResult, ToolCallStart
from box_agent.runtime import run_agent_loop
from box_agent.schema import FunctionCall, LLMResponse, Message, StreamEvent, ToolCall
from box_agent.tools import mcp_loader
from box_agent.tools.base import Tool, ToolResult
from box_agent.tools.mcp_tool_catalog import MCPToolCatalog, get_mcp_tool_catalog
from box_agent.tools.mcp_tool_search import MCPToolExposureManager, ToolSearchTool
from box_agent.tools.setup import (
    register_mcp_tools,
    sync_mcp_tool_list,
    sync_mcp_tools,
)


class FakeMCPTool(Tool):
    def __init__(
        self,
        name: str,
        server_name: str,
        description: str = "",
        *,
        always_load: bool = False,
        parameters: dict | None = None,
    ) -> None:
        self._name = name
        self._server_name = server_name
        self._description = description
        self._always_load = always_load
        self._parameters = parameters or {"type": "object", "properties": {}}
        self._mcp_generation = 0
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict:
        return self._parameters

    @property
    def mcp_tool_id(self) -> str:
        return f"mcp:{self._server_name}/{self._name}"

    @property
    def server_name(self) -> str:
        return self._server_name

    @property
    def mcp_always_load(self) -> bool:
        return self._always_load

    @property
    def mcp_generation(self) -> int:
        return self._mcp_generation

    async def execute(self, **kwargs) -> ToolResult:
        self.calls += 1
        return ToolResult(success=True, content=json.dumps(kwargs))


class FakeCoreTool(Tool):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Stable core tool"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, content=json.dumps(kwargs))


class MockLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = responses
        self._index = 0
        self.offered_names: list[list[str]] = []

    async def generate_stream(self, messages, tools=None, **_):
        self.offered_names.append([tool.name for tool in tools or []])
        response = self._responses[self._index]
        self._index += 1
        if response.content:
            yield StreamEvent(type="text", delta=response.content)
        yield StreamEvent(
            type="finish",
            finish_reason=response.finish_reason,
            usage=response.usage,
            tool_calls=response.tool_calls,
        )


async def _collect(source):
    return [event async for event in source]


@pytest.mark.asyncio
async def test_search_activates_only_the_calling_session() -> None:
    catalog = MCPToolCatalog()
    schema = {
        "type": "object",
        "properties": {"customer_id": {"type": "string"}},
    }
    tool = FakeMCPTool(
        "find_customer",
        "crm",
        "Find a customer account",
        parameters=schema,
    )
    catalog.replace_server("crm", [tool])
    first_activated = OrderedDict()
    second_activated = OrderedDict()
    first = MCPToolExposureManager(catalog, first_activated)
    second = MCPToolExposureManager(catalog, second_activated)
    search = ToolSearchTool(catalog, first_activated)

    assert first.prepare_tools([]).tools == []
    result = await search.execute(query="customer")

    assert result.success
    assert json.loads(result.content)["activated"][0]["name"] == "find_customer"
    exposed = first.prepare_tools([]).tools
    assert [item.name for item in exposed] == ["find_customer"]
    assert exposed[0].parameters == schema
    assert second.prepare_tools([]).tools == []


def test_always_load_is_visible_without_activation() -> None:
    catalog = MCPToolCatalog()
    tool = FakeMCPTool("health_check", "ops", always_load=True)
    catalog.replace_server("ops", [tool])
    manager = MCPToolExposureManager(catalog, OrderedDict())

    exposure = manager.prepare_tools([tool])

    assert exposure.offered_names == frozenset({"health_check"})


def test_catalog_retains_real_schema_without_putting_tool_in_candidates() -> None:
    catalog = MCPToolCatalog()
    schema = {
        "type": "object",
        "properties": {"customer_id": {"type": "string"}},
        "required": ["customer_id"],
    }
    tool = FakeMCPTool("lookup", "crm", parameters=schema)
    catalog.replace_server("crm", [tool])
    entry = catalog.get("mcp:crm/lookup")

    assert entry is not None
    assert entry.tool.parameters == schema
    assert MCPToolExposureManager(catalog, OrderedDict()).prepare_tools([]).tools == []


@pytest.mark.asyncio
async def test_search_activates_exactly_requested_hits_without_small_cap() -> None:
    catalog = MCPToolCatalog()
    names = [f"report_{index:02d}" for index in range(12)]
    for name in names:
        catalog.replace_server(
            name,
            [FakeMCPTool(name, name, "shared reporting capability")],
        )
    activated = OrderedDict()
    search = ToolSearchTool(catalog, activated)

    result = await search.execute(query="reporting", top_k=10)
    payload = json.loads(result.content)
    exposed = MCPToolExposureManager(catalog, activated).prepare_tools([])

    assert search.parameters["properties"]["top_k"]["default"] == 1
    assert "maximum" not in search.parameters["properties"]["top_k"]
    assert [item["name"] for item in payload["activated"]] == names[:10]
    assert [tool.name for tool in exposed.tools] == names[:10]
    assert set(names[10:]).isdisjoint(exposed.offered_names)
    assert "Only these hits" in payload["notice"]


@pytest.mark.asyncio
async def test_search_finds_multiple_exact_tool_names_inside_compound_query() -> None:
    catalog = MCPToolCatalog()
    navigate = FakeMCPTool(
        "browser_navigate",
        "playwright",
        "Navigate to a URL",
    )
    snapshot = FakeMCPTool(
        "browser_snapshot",
        "playwright",
        "Capture the current page snapshot",
    )
    catalog.replace_server("playwright", [navigate, snapshot])
    activated = OrderedDict()

    result = await ToolSearchTool(catalog, activated).execute(
        query=(
            "playwright browser_navigate navigate URL and browser_snapshot "
            "get exact page text"
        ),
        server_name="playwright",
        top_k=10,
    )
    payload = json.loads(result.content)

    assert result.success is True
    assert {item["name"] for item in payload["activated"]} == {
        "browser_navigate",
        "browser_snapshot",
    }


@pytest.mark.asyncio
async def test_duplicate_model_names_are_reported_but_not_activated() -> None:
    catalog = MCPToolCatalog()
    first = FakeMCPTool("lookup", "crm", "CRM lookup")
    second = FakeMCPTool("lookup", "erp", "ERP lookup")
    catalog.replace_server("crm", [first])
    catalog.replace_server("erp", [second])
    activated = OrderedDict()

    result = await ToolSearchTool(catalog, activated).execute(
        query="lookup",
        top_k=2,
    )
    payload = json.loads(result.content)

    assert result.success is False
    assert len(payload["conflicts"]) == 2
    assert payload["activated"] == []
    assert activated == OrderedDict()


@pytest.mark.asyncio
async def test_mcp_tool_cannot_replace_same_named_stable_core_tool() -> None:
    catalog = MCPToolCatalog()
    core_bash = FakeCoreTool("bash")
    remote_bash = FakeMCPTool("bash", "untrusted", always_load=True)
    catalog.replace_server("untrusted", [remote_bash])
    activated = OrderedDict()
    manager = MCPToolExposureManager(catalog, activated)
    search = ToolSearchTool(
        catalog,
        activated,
        protected_names_provider=lambda: frozenset({"bash", "tool_search"}),
    )

    result = await search.execute(query="bash")
    payload = json.loads(result.content)
    exposure = manager.prepare_tools([core_bash])

    assert result.success is False
    assert payload["activated"] == []
    assert payload["conflicts"][0]["error"] == "conflicts with stable core tool"
    assert activated == OrderedDict()
    assert exposure.tools == [core_bash]
    assert exposure.mcp_generations == {}


@pytest.mark.asyncio
async def test_search_waits_for_initial_catalog_discovery() -> None:
    catalog = MCPToolCatalog()
    activated = OrderedDict()
    search = ToolSearchTool(catalog, activated, readiness_timeout=1.0)
    catalog.mark_loading()

    pending = asyncio.create_task(search.execute(query="lookup"))
    await asyncio.sleep(0)
    assert not pending.done()

    catalog.replace_server("crm", [FakeMCPTool("lookup", "crm", "Lookup")])
    catalog.mark_ready()
    result = await pending

    assert result.success is True
    assert list(activated) == ["mcp:crm/lookup"]


@pytest.mark.asyncio
async def test_search_reports_loading_instead_of_false_empty_result() -> None:
    catalog = MCPToolCatalog()
    catalog.mark_loading()
    result = await ToolSearchTool(
        catalog,
        OrderedDict(),
        readiness_timeout=0.001,
    ).execute(query="lookup")
    payload = json.loads(result.content)

    assert result.success is False
    assert payload["state"] == "catalog_loading"
    assert payload["activated"] == []
    catalog.mark_ready()


@pytest.mark.asyncio
async def test_reconnect_generation_invalidates_session_activation() -> None:
    catalog = MCPToolCatalog()
    original = FakeMCPTool("lookup", "crm", "Lookup")
    catalog.replace_server("crm", [original])
    activated = OrderedDict()
    manager = MCPToolExposureManager(catalog, activated)
    await ToolSearchTool(catalog, activated).execute(query="lookup")
    offered = manager.prepare_tools([original])

    replacement = FakeMCPTool("lookup", "crm", "Lookup updated")
    catalog.replace_server("crm", [replacement])

    assert manager.prepare_tools([replacement]).tools == []
    assert "changed" in manager.validate_call(
        "lookup",
        offered.mcp_generations["lookup"],
    )


@pytest.mark.asyncio
async def test_core_rejects_hidden_tool_not_offered_in_step() -> None:
    catalog = MCPToolCatalog()
    hidden = FakeMCPTool("lookup", "crm", "Lookup")
    catalog.replace_server("crm", [hidden])
    activated = OrderedDict()
    manager = MCPToolExposureManager(catalog, activated)
    search = ToolSearchTool(catalog, activated)
    llm = MockLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="hidden-call",
                        type="function",
                        function=FunctionCall(name="lookup", arguments={}),
                    )
                ],
                finish_reason="tool",
            ),
            LLMResponse(content="done", tool_calls=[], finish_reason="stop"),
        ]
    )

    events = await _collect(
        run_agent_loop(
            llm=llm,
            messages=[Message(role="system", content="test")],
            tools={"lookup": hidden, "tool_search": search},
            max_steps=2,
            tool_exposure_manager=manager,
        )
    )

    result = next(event for event in events if isinstance(event, ToolCallResult))
    assert hidden.calls == 0
    assert "not offered" in (result.error or "")
    assert llm.offered_names[0] == ["tool_search"]


@pytest.mark.asyncio
async def test_search_then_real_tool_is_exposed_on_next_step() -> None:
    catalog = MCPToolCatalog()
    tool = FakeMCPTool("lookup", "crm", "Lookup")
    catalog.replace_server("crm", [tool])
    activated = OrderedDict()
    manager = MCPToolExposureManager(catalog, activated)
    search = ToolSearchTool(catalog, activated)
    llm = MockLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="search-call",
                        type="function",
                        function=FunctionCall(
                            name="tool_search",
                            arguments={"query": "lookup"},
                        ),
                    )
                ],
                finish_reason="tool",
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="real-call",
                        type="function",
                        function=FunctionCall(name="lookup", arguments={}),
                    )
                ],
                finish_reason="tool",
            ),
            LLMResponse(content="done", tool_calls=[], finish_reason="stop"),
        ]
    )

    events = await _collect(
        run_agent_loop(
            llm=llm,
            messages=[Message(role="system", content="test")],
            tools={"tool_search": search},
            max_steps=3,
            tool_exposure_manager=manager,
        )
    )

    assert llm.offered_names[0] == ["tool_search"]
    assert set(llm.offered_names[1]) == {"lookup", "tool_search"}
    assert tool.calls == 1
    real_start = next(
        event
        for event in events
        if isinstance(event, ToolCallStart) and event.tool_call_id == "real-call"
    )
    real_result = next(
        event
        for event in events
        if isinstance(event, ToolCallResult) and event.tool_call_id == "real-call"
    )
    assert real_start.tool_id == "mcp:crm/lookup"
    assert real_start.server_name == "crm"
    assert real_result.tool_id == "mcp:crm/lookup"
    assert real_result.server_name == "crm"


@pytest.mark.asyncio
async def test_agent_wires_search_and_child_inheritance_to_session_visibility(
    tmp_path,
) -> None:
    catalog = get_mcp_tool_catalog()
    catalog.clear()
    tool = FakeMCPTool("lookup", "crm", "Lookup")
    catalog.replace_server("crm", [tool])
    try:
        agent = Agent(
            llm_client=object(),
            system_prompt="test",
            tools=[tool],
            workspace_dir=str(tmp_path),
            deferred_mcp_loading_enabled=True,
        )

        assert "tool_search" in agent.tools
        assert "lookup" not in agent.tools
        assert "lookup" not in agent._inherited_tools()
        assert "tool_search" not in agent._inherited_tools()

        result = await agent.tools["tool_search"].execute(query="lookup")

        assert result.success
        assert "lookup" in agent._inherited_tools()
    finally:
        catalog.clear()


def test_agent_explicit_legacy_mode_keeps_existing_eager_behavior(tmp_path) -> None:
    tool = FakeMCPTool("lookup", "crm", "Lookup")
    agent = Agent(
        llm_client=object(),
        system_prompt="test",
        tools=[tool],
        workspace_dir=str(tmp_path),
        deferred_mcp_loading_enabled=False,
    )

    assert "tool_search" not in agent.tools
    assert agent._inherited_tools()["lookup"] is tool


def test_late_mcp_registration_cannot_replace_session_search_control() -> None:
    catalog = MCPToolCatalog()
    search = ToolSearchTool(catalog, OrderedDict())
    remote_collision = FakeMCPTool("tool_search", "remote")
    tool_map = {"tool_search": search}

    register_mcp_tools(tool_map, [remote_collision])

    assert tool_map["tool_search"] is search


@pytest.mark.asyncio
async def test_loader_catalogs_always_load_server_tools(tmp_path, monkeypatch) -> None:
    await mcp_loader.cleanup_mcp_connections()
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "ops": {
                        "command": "fake",
                        "alwaysLoad": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    async def fake_connect(connection) -> bool:
        connection.tools = [
            FakeMCPTool(
                "health_check",
                connection.name,
                always_load=connection.always_load,
            )
        ]
        return True

    monkeypatch.setattr(mcp_loader.MCPServerConnection, "connect", fake_connect)
    try:
        tools = await mcp_loader.load_mcp_tools_async(str(config_path))
        entry = get_mcp_tool_catalog().get("mcp:ops/health_check")

        assert [tool.name for tool in tools] == ["health_check"]
        assert entry is not None
        assert entry.always_load is True
        assert entry.generation == 1

        reconnect_result = await mcp_loader.reconnect_mcp_server("ops")
        replacement = get_mcp_tool_catalog().get("mcp:ops/health_check")
        assert reconnect_result["success"] is True
        assert replacement is not None
        assert replacement.generation == 2

        await mcp_loader.disconnect_mcp_server("ops")
        assert get_mcp_tool_catalog().get("mcp:ops/health_check") is None
    finally:
        await mcp_loader.cleanup_mcp_connections()


@pytest.mark.asyncio
async def test_reconnect_reports_loading_until_replacement_catalog_is_ready(
    tmp_path,
    monkeypatch,
) -> None:
    await mcp_loader.cleanup_mcp_connections()
    catalog = get_mcp_tool_catalog()
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps({"mcpServers": {"crm": {"command": "fake"}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(mcp_loader, "_mcp_config_path", str(config_path))
    connect_started = asyncio.Event()
    release_connect = asyncio.Event()

    async def blocked_connect(connection) -> bool:
        connect_started.set()
        await release_connect.wait()
        connection.tools = [FakeMCPTool("lookup", connection.name, "Lookup")]
        return True

    monkeypatch.setattr(mcp_loader.MCPServerConnection, "connect", blocked_connect)
    reconnect = asyncio.create_task(mcp_loader.reconnect_mcp_server("crm"))
    try:
        await connect_started.wait()
        loading_result = await ToolSearchTool(
            catalog,
            OrderedDict(),
            readiness_timeout=0.001,
        ).execute(query="lookup")
        assert loading_result.success is False
        assert json.loads(loading_result.content)["state"] == "catalog_loading"

        release_connect.set()
        reconnect_result = await reconnect
        assert reconnect_result["success"] is True
        ready_result = await ToolSearchTool(
            catalog,
            OrderedDict(),
            readiness_timeout=0.001,
        ).execute(query="lookup")
        assert ready_result.success is True
        assert json.loads(ready_result.content)["activated"][0]["name"] == "lookup"
    finally:
        release_connect.set()
        if not reconnect.done():
            await reconnect
        await mcp_loader.cleanup_mcp_connections()


@pytest.mark.asyncio
async def test_reconnect_failure_releases_catalog_loading_state(
    tmp_path,
    monkeypatch,
) -> None:
    await mcp_loader.cleanup_mcp_connections()
    catalog = get_mcp_tool_catalog()
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps({"mcpServers": {"crm": {"command": "fake"}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(mcp_loader, "_mcp_config_path", str(config_path))

    async def failed_connect(_connection) -> bool:
        raise RuntimeError("connect failed")

    monkeypatch.setattr(mcp_loader.MCPServerConnection, "connect", failed_connect)
    try:
        result = await mcp_loader.reconnect_mcp_server("crm")
        assert result["success"] is False
        assert catalog.loading is False
        assert mcp_loader.is_mcp_loading() is False
    finally:
        await mcp_loader.cleanup_mcp_connections()


@pytest.mark.asyncio
async def test_same_server_reconnects_remain_serialized_with_queued_waiters(
    tmp_path,
    monkeypatch,
) -> None:
    await mcp_loader.cleanup_mcp_connections()
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps({"mcpServers": {"crm": {"command": "fake"}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(mcp_loader, "_mcp_config_path", str(config_path))
    release_first = asyncio.Event()
    release_second = asyncio.Event()
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    third_started = asyncio.Event()
    call_count = 0

    async def serialized_connect(connection) -> bool:
        nonlocal call_count
        call_count += 1
        current = call_count
        if current == 1:
            first_started.set()
            await release_first.wait()
        elif current == 2:
            second_started.set()
            await release_second.wait()
        else:
            third_started.set()
        connection.tools = [FakeMCPTool("lookup", connection.name, "Lookup")]
        return True

    monkeypatch.setattr(
        mcp_loader.MCPServerConnection,
        "connect",
        serialized_connect,
    )
    first = asyncio.create_task(mcp_loader.reconnect_mcp_server("crm"))
    second = asyncio.create_task(mcp_loader.reconnect_mcp_server("crm"))
    third: asyncio.Task | None = None
    try:
        await first_started.wait()
        release_first.set()
        await first
        await second_started.wait()
        third = asyncio.create_task(mcp_loader.reconnect_mcp_server("crm"))
        await asyncio.sleep(0)
        assert third_started.is_set() is False

        release_second.set()
        await asyncio.gather(second, third)
        assert third_started.is_set() is True
    finally:
        release_first.set()
        release_second.set()
        for task in (first, second, third):
            if task is not None and not task.done():
                await task
        await mcp_loader.cleanup_mcp_connections()


def test_sync_mcp_registrations_restore_stable_fallbacks() -> None:
    stable = FakeCoreTool("lookup")
    crm = FakeMCPTool("lookup", "crm")
    erp = FakeMCPTool("lookup", "erp")

    tool_map = {"lookup": stable}
    map_fallbacks: dict[str, Tool] = {}
    sync_mcp_tools(tool_map, [crm, erp], map_fallbacks)
    assert tool_map["lookup"] is erp
    sync_mcp_tools(tool_map, [crm], map_fallbacks)
    assert tool_map["lookup"] is crm
    sync_mcp_tools(tool_map, [], map_fallbacks)
    assert tool_map["lookup"] is stable

    tool_list: list[Tool] = [stable]
    list_fallbacks: dict[str, Tool] = {}
    sync_mcp_tool_list(tool_list, [crm], list_fallbacks)
    assert tool_list == [crm]
    sync_mcp_tool_list(tool_list, [], list_fallbacks)
    assert tool_list == [stable]
