from __future__ import annotations

from pathlib import Path

from box_agent.tools.base import Tool, ToolResult
from box_agent.tools.skill_loader import SkillLoader
from box_agent.tools.sub_agent_capabilities import (
    BATCH_FILES_MAX_FILES,
    CapabilityFailure,
    CapabilityResolver,
    DelegationSpec,
    ResolvedCapabilityBundle,
    parse_delegation_spec,
)


class NamedTool(Tool):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._name

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, content=self._name)


class NamedMcpTool(NamedTool):
    def __init__(self, name: str, server_name: str):
        super().__init__(name)
        self._server_name = server_name

    @property
    def server_name(self) -> str:
        return self._server_name


def _parse(**overrides) -> DelegationSpec | CapabilityFailure:
    values = {
        "task": "Inspect the files",
        "capabilities": {"required_tools": ["read_file"]},
    }
    values.update(overrides)
    return parse_delegation_spec(**values)


def _write_skill(
    root: Path,
    name: str,
    *,
    required: list[str] | None = None,
    related: list[str] | None = None,
    allowed_tools: list[str] | None = None,
) -> None:
    skill_dir = root / name
    skill_dir.mkdir()
    lines = ["---", f"name: {name}", f"description: {name} description"]
    if required is not None:
        lines.append(f"required_skills: [{', '.join(required)}]")
    if related is not None:
        lines.append(f"related_skills: [{', '.join(related)}]")
    if allowed_tools is not None:
        lines.append("allowed-tools:")
        lines.extend(f"  - {tool_name}" for tool_name in allowed_tools)
    lines.extend(["---", "", f"Instructions for {name}."])
    (skill_dir / "SKILL.md").write_text("\n".join(lines), encoding="utf-8")


def test_minimal_spec_applies_safe_defaults_and_stable_order() -> None:
    parsed = _parse(
        capabilities={
            "required_tools": ["read_file", "read_file"],
            "optional_tools": ["web_search", "get_skill", "web_search"],
            "skills": ["zeta", "alpha", "zeta"],
        }
    )

    assert isinstance(parsed, DelegationSpec)
    assert parsed.strategy == "general_loop"
    assert parsed.required_tools == ("read_file",)
    assert parsed.optional_tools == ("get_skill", "web_search")
    assert parsed.skill_names == ("alpha", "zeta")
    assert parsed.constraints.read_only is True
    assert parsed.constraints.network is False
    assert parsed.constraints.external_side_effect is False
    assert parsed.constraints.write_scope is None
    assert parsed.budget.max_steps == 60
    assert parsed.budget.max_tool_calls == 32
    assert "execution.strategy" in parsed.defaults_applied
    assert "constraints.read_only" in parsed.defaults_applied


def test_general_loop_budget_uses_configured_caps() -> None:
    parsed = _parse(
        budget={"max_steps": 99, "max_tool_calls": 99},
        general_max_steps=20,
        general_max_tool_calls=30,
    )

    assert isinstance(parsed, DelegationSpec)
    assert parsed.budget.max_steps == 20
    assert parsed.budget.max_tool_calls == 30


def test_invalid_new_style_spec_returns_retryable_field_diagnostics() -> None:
    parsed = parse_delegation_spec(
        task="",
        capabilities={},
        execution={"strategy": "unknown", "extra": True},
    )

    assert isinstance(parsed, CapabilityFailure)
    payload = parsed.to_dict()
    assert payload["code"] == "INVALID_DELEGATION_SPEC"
    assert payload["retryable"] is True
    assert "task" in payload["invalid_fields"]
    assert "capabilities.required_tools" in payload["invalid_fields"]
    assert "execution.strategy" in payload["invalid_fields"]
    assert "execution.extra" in payload["invalid_fields"]
    assert payload["minimal_valid_example"]["capabilities"]["required_tools"] == [
        "read_file"
    ]


def test_batch_files_requires_read_file_and_normalizes_files() -> None:
    parsed = _parse(
        execution={"strategy": "batch_files"},
        inputs={"files": ["b.md", "a.md", "b.md"]},
        budget={"max_steps": 9, "max_tool_calls": 20},
    )

    assert isinstance(parsed, DelegationSpec)
    assert parsed.inputs["files"] == ["a.md", "b.md"]
    assert parsed.budget.max_steps == 1
    assert parsed.budget.max_tool_calls == 2

    insufficient_budget = _parse(
        execution={"strategy": "batch_files"},
        inputs={"files": ["a.md", "b.md"]},
        budget={"max_tool_calls": 1},
    )
    assert isinstance(insufficient_budget, CapabilityFailure)
    assert "budget.max_tool_calls" in insufficient_budget.invalid_fields

    invalid_tool = _parse(
        execution={"strategy": "batch_files"},
        capabilities={"required_tools": ["bash"]},
        inputs={"files": ["a.md"]},
    )
    assert isinstance(invalid_tool, CapabilityFailure)
    assert "capabilities.required_tools" in invalid_tool.invalid_fields

    too_many = _parse(
        execution={"strategy": "batch_files"},
        inputs={"files": [f"{index}.md" for index in range(BATCH_FILES_MAX_FILES + 1)]},
    )
    assert isinstance(too_many, CapabilityFailure)
    assert "inputs.files" in too_many.invalid_fields


def test_required_builtin_missing_is_not_found_even_while_mcp_loads() -> None:
    spec = _parse()
    assert isinstance(spec, DelegationSpec)

    result = CapabilityResolver().resolve(
        spec,
        parent_tools={},
        capability_state="loading",
    )

    assert isinstance(result, CapabilityFailure)
    assert result.code == "REQUIRED_TOOL_NOT_FOUND"
    assert result.retryable is False


def test_unknown_required_tool_distinguishes_mcp_loading_and_ready() -> None:
    spec = _parse(capabilities={"required_tools": ["mcp_future_tool"]})
    assert isinstance(spec, DelegationSpec)

    loading = CapabilityResolver().resolve(
        spec,
        parent_tools={},
        capability_state={"state": "loading"},
    )
    ready = CapabilityResolver().resolve(
        spec,
        parent_tools={},
        capability_state="ready",
    )

    assert isinstance(loading, CapabilityFailure)
    assert loading.code == "REQUIRED_TOOL_NOT_READY"
    assert loading.pending_source == "mcp"
    assert loading.retryable is True
    assert isinstance(ready, CapabilityFailure)
    assert ready.code == "REQUIRED_TOOL_NOT_FOUND"
    assert ready.retryable is False


def test_constraints_reject_required_writes_and_network_tools() -> None:
    write_spec = _parse(capabilities={"required_tools": ["write_file"]})
    assert isinstance(write_spec, DelegationSpec)
    write_result = CapabilityResolver().resolve(
        write_spec,
        parent_tools={"write_file": NamedTool("write_file")},
    )
    assert isinstance(write_result, CapabilityFailure)
    assert write_result.code == "CAPABILITY_CONSTRAINT_CONFLICT"
    assert write_result.details["denied_reason"] == "read_only"

    network_spec = _parse(capabilities={"required_tools": ["web_search"]})
    assert isinstance(network_spec, DelegationSpec)
    network_result = CapabilityResolver().resolve(
        network_spec,
        parent_tools={"web_search": NamedTool("web_search")},
    )
    assert isinstance(network_result, CapabilityFailure)
    assert network_result.details["denied_reason"] == "network_disabled"


def test_scoped_web_research_write_constraints_resolve_minimum_tools() -> None:
    spec = _parse(
        capabilities={"required_tools": ["web_search", "write_file"]},
        constraints={
            "read_only": False,
            "network": True,
            "write_scope": ["research/dim01.md"],
            "external_side_effect": False,
        },
        budget={"max_steps": 12, "max_tool_calls": 25},
    )
    assert isinstance(spec, DelegationSpec)

    result = CapabilityResolver().resolve(
        spec,
        parent_tools={
            "web_search": NamedTool("web_search"),
            "write_file": NamedTool("write_file"),
        },
    )

    assert isinstance(result, ResolvedCapabilityBundle)
    assert result.resolved_tool_names == ("web_search", "write_file")
    assert result.spec.constraints.write_scope == ("research/dim01.md",)
    assert result.spec.budget.max_steps == 12
    assert result.spec.budget.max_tool_calls == 25


def test_playwright_capabilities_are_explicit_and_constraint_aware() -> None:
    navigate = NamedMcpTool("browser_navigate", "playwright")
    navigate_spec = _parse(
        capabilities={"required_tools": ["browser_navigate"]},
        constraints={"network": True},
    )
    assert isinstance(navigate_spec, DelegationSpec)
    navigate_result = CapabilityResolver().resolve(
        navigate_spec,
        parent_tools={"browser_navigate": navigate},
    )
    assert isinstance(navigate_result, ResolvedCapabilityBundle)
    assert navigate_result.resolved_tool_names == ("browser_navigate",)

    run_code = NamedMcpTool("browser_run_code", "playwright")
    guarded_spec = _parse(
        capabilities={"required_tools": ["browser_run_code"]},
        constraints={"network": True},
    )
    assert isinstance(guarded_spec, DelegationSpec)
    guarded_result = CapabilityResolver().resolve(
        guarded_spec,
        parent_tools={"browser_run_code": run_code},
    )
    assert isinstance(guarded_result, CapabilityFailure)
    assert guarded_result.details["denied_reason"] == "external_side_effect_disabled"

    granted_spec = _parse(
        capabilities={"required_tools": ["browser_run_code"]},
        constraints={"network": True, "external_side_effect": True},
    )
    assert isinstance(granted_spec, DelegationSpec)
    granted_result = CapabilityResolver().resolve(
        granted_spec,
        parent_tools={"browser_run_code": run_code},
    )
    assert isinstance(granted_result, ResolvedCapabilityBundle)
    assert granted_result.resolved_tool_names == ("browser_run_code",)


def test_required_sub_agent_is_always_rejected() -> None:
    spec = _parse(capabilities={"required_tools": ["sub_agent"]})
    assert isinstance(spec, DelegationSpec)

    result = CapabilityResolver().resolve(
        spec,
        parent_tools={"sub_agent": NamedTool("sub_agent")},
    )

    assert isinstance(result, CapabilityFailure)
    assert result.code == "CAPABILITY_CONSTRAINT_CONFLICT"
    assert result.details["denied_reason"] == "recursive_delegation_disabled"


def test_optional_tools_are_recorded_but_do_not_block() -> None:
    spec = _parse(
        capabilities={
            "required_tools": ["read_file"],
            "optional_tools": ["write_file", "missing_optional"],
        }
    )
    assert isinstance(spec, DelegationSpec)
    result = CapabilityResolver().resolve(
        spec,
        parent_tools={
            "read_file": NamedTool("read_file"),
            "write_file": NamedTool("write_file"),
        },
        capability_state="ready",
    )

    assert isinstance(result, ResolvedCapabilityBundle)
    assert result.resolved_tool_names == ("read_file",)
    assert result.denied_tools == (
        {"name": "missing_optional", "origin": "optional", "reason": "not_found"},
        {"name": "write_file", "origin": "optional", "reason": "read_only"},
    )


def test_skill_dependencies_expand_allowed_tools_without_related_skills(tmp_path: Path) -> None:
    _write_skill(tmp_path, "base", allowed_tools=["read_file"])
    _write_skill(tmp_path, "related", allowed_tools=["write_file"])
    _write_skill(
        tmp_path,
        "selected",
        required=["base"],
        related=["related"],
        allowed_tools=["web_search"],
    )
    loader = SkillLoader(tmp_path)
    loader.discover_skills()
    spec = _parse(
        capabilities={"required_tools": ["read_file"], "skills": ["selected"]},
        constraints={
            "read_only": True,
            "network": True,
            "write_scope": None,
            "external_side_effect": False,
        },
    )
    assert isinstance(spec, DelegationSpec)

    result = CapabilityResolver().resolve(
        spec,
        parent_tools={
            "read_file": NamedTool("read_file"),
            "web_search": NamedTool("web_search"),
            "write_file": NamedTool("write_file"),
        },
        skill_loader=loader,
    )

    assert isinstance(result, ResolvedCapabilityBundle)
    assert result.resolved_skill_names == ("base", "selected")
    assert "related" not in result.resolved_skill_names
    assert result.resolved_tool_names == ("read_file", "web_search")
    assert result.skill_added_tools == ("web_search",)


def test_declared_skill_requires_a_live_provider() -> None:
    spec = _parse(
        capabilities={"required_tools": ["read_file"], "skills": ["selected"]}
    )
    assert isinstance(spec, DelegationSpec)

    result = CapabilityResolver().resolve(
        spec,
        parent_tools={"read_file": NamedTool("read_file")},
        skill_loader=None,
    )

    assert isinstance(result, CapabilityFailure)
    assert result.code == "SKILL_PROVIDER_UNAVAILABLE"


def test_required_skill_cycle_fails_deterministically(tmp_path: Path) -> None:
    _write_skill(tmp_path, "alpha", required=["beta"])
    _write_skill(tmp_path, "beta", required=["alpha"])
    loader = SkillLoader(tmp_path)
    loader.discover_skills()
    spec = _parse(
        capabilities={"required_tools": ["read_file"], "skills": ["alpha"]}
    )
    assert isinstance(spec, DelegationSpec)

    result = CapabilityResolver().resolve(
        spec,
        parent_tools={"read_file": NamedTool("read_file")},
        skill_loader=loader,
    )

    assert isinstance(result, CapabilityFailure)
    assert result.code == "SKILL_DEPENDENCY_CYCLE"
    assert result.details["cycle"] == ["alpha", "beta", "alpha"]


def test_disabled_and_broken_skills_fail_before_child_start(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write_skill(skills_dir, "disabled")
    broken_dir = skills_dir / "broken"
    broken_dir.mkdir()
    (broken_dir / "SKILL.md").write_text("not valid frontmatter", encoding="utf-8")
    settings = tmp_path / "skill-settings.json"
    settings.write_text('{"disabledSkillNames":["disabled"]}', encoding="utf-8")
    loader = SkillLoader(skills_dir, skill_settings_path=settings)
    loader.discover_skills()

    for skill_name, code in (("disabled", "SKILL_DISABLED"), ("broken", "SKILL_BROKEN")):
        spec = _parse(
            capabilities={
                "required_tools": ["read_file"],
                "skills": [skill_name],
            }
        )
        assert isinstance(spec, DelegationSpec)
        result = CapabilityResolver().resolve(
            spec,
            parent_tools={"read_file": NamedTool("read_file")},
            skill_loader=loader,
        )
        assert isinstance(result, CapabilityFailure)
        assert result.code == code
