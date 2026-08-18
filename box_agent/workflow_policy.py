"""Stable contract between the agent kernel and optional workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .tools.base import ToolResult


@dataclass(frozen=True, slots=True)
class WorkflowCheckpointUpdate:
    """A workflow checkpoint plus evidence recovered from persisted state."""

    text: str
    changed: bool
    recovered_evidence_urls: frozenset[str] = frozenset()


class WorkflowPolicy(Protocol):
    """Host-neutral workflow hooks consumed by the agent kernel.

    Implementations may keep per-run state, but the kernel only sees this
    contract and does not import or branch on a concrete workflow.
    """

    kind: str
    checkpoint_injection_id: str
    evidence_read_batch_size: int

    def build_checkpoint(self) -> str | None:
        """Re-derive the workflow checkpoint from canonical state."""
        ...

    def update_checkpoint(
        self,
        checkpoint_text: str,
    ) -> WorkflowCheckpointUpdate: ...

    def plan_scope_error(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str | None: ...

    def tool_call_error(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        verified_evidence_urls: set[str],
        parallel: bool = False,
    ) -> str | None: ...

    def record_tool_result(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: ToolResult,
        *,
        executed: bool = True,
    ) -> None: ...

    def exempts_tool_budget(self, tool_name: str) -> bool: ...

    def uses_evidence_read_budget(self, tool_name: str) -> bool: ...

    def is_direct_evidence_read_tool(self, tool_name: str) -> bool: ...

    def direct_evidence_url(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: ToolResult,
    ) -> str | None: ...

    def allows_completion_continuation(self) -> bool: ...

    def suppresses_generic_final_summary(self) -> bool: ...
