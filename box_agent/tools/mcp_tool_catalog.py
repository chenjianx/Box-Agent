"""Process-wide catalog of tools discovered from connected MCP servers.

The catalog owns discovery metadata only.  Which catalog entries are visible to
an LLM is decided per Agent session by :mod:`mcp_tool_search`.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable
from dataclasses import dataclass
from threading import RLock

from .base import Tool


def _normalize(value: str) -> str:
    value = value.strip().lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", value)


@dataclass(frozen=True, slots=True)
class MCPToolEntry:
    tool_id: str
    model_name: str
    server_name: str
    raw_tool_name: str
    description: str
    tool: Tool
    generation: int
    always_load: bool
    name_conflict: bool = False


class MCPToolCatalog:
    """Thread-safe current snapshot of connected MCP tools."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._entries: dict[str, MCPToolEntry] = {}
        self._server_generations: dict[str, int] = {}
        self._loading = False
        self._refreshing_servers: set[str] = set()
        self._ready_event: asyncio.Event | None = None

    def _has_pending_discovery(self) -> bool:
        return self._loading or bool(self._refreshing_servers)

    def _ensure_pending_event(self) -> None:
        if self._ready_event is None or self._ready_event.is_set():
            self._ready_event = asyncio.Event()

    def mark_loading(self) -> None:
        """Mark initial discovery as incomplete without resetting live waiters."""
        with self._lock:
            if self._loading:
                return
            was_pending = self._has_pending_discovery()
            self._loading = True
            if not was_pending:
                self._ensure_pending_event()

    def mark_ready(self) -> None:
        """Publish discovery completion and release pending catalog searches."""
        with self._lock:
            self._loading = False
            ready_event = (
                self._ready_event if not self._has_pending_discovery() else None
            )
        if ready_event is not None:
            ready_event.set()

    def mark_server_loading(self, server_name: str) -> None:
        """Mark one hot-reloading server unavailable for catalog searches."""
        with self._lock:
            if server_name in self._refreshing_servers:
                return
            was_pending = self._has_pending_discovery()
            self._refreshing_servers.add(server_name)
            if not was_pending:
                self._ensure_pending_event()

    def mark_server_ready(self, server_name: str) -> None:
        """Finish one hot-reload and release waiters when discovery is stable."""
        with self._lock:
            self._refreshing_servers.discard(server_name)
            ready_event = (
                self._ready_event if not self._has_pending_discovery() else None
            )
        if ready_event is not None:
            ready_event.set()

    @property
    def loading(self) -> bool:
        with self._lock:
            return self._has_pending_discovery()

    async def wait_until_ready(self, timeout: float) -> bool:
        """Wait for active discovery, returning False on a bounded timeout."""
        with self._lock:
            if not self._has_pending_discovery():
                return True
            ready_event = self._ready_event
        if ready_event is None:
            return True
        try:
            await asyncio.wait_for(ready_event.wait(), timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError):
            return False
        return True

    def replace_server(self, server_name: str, tools: Iterable[Tool]) -> int:
        """Replace one server snapshot and return its new generation."""
        with self._lock:
            generation = self._server_generations.get(server_name, 0) + 1
            self._server_generations[server_name] = generation
            self._entries = {
                tool_id: entry
                for tool_id, entry in self._entries.items()
                if entry.server_name != server_name
            }
            for tool in tools:
                raw_name = tool.name
                tool_id = f"mcp:{server_name}/{raw_name}"
                # MCPTool exposes these attributes.  Keeping the catalog
                # tolerant of Tool doubles makes focused tests inexpensive.
                tool._mcp_generation = generation
                self._entries[tool_id] = MCPToolEntry(
                    tool_id=tool_id,
                    model_name=raw_name,
                    server_name=server_name,
                    raw_tool_name=raw_name,
                    description=tool.description,
                    tool=tool,
                    generation=generation,
                    always_load=bool(getattr(tool, "mcp_always_load", False)),
                )
            self._rebuild_conflicts()
            return generation

    def remove_server(self, server_name: str) -> None:
        with self._lock:
            self._entries = {
                tool_id: entry
                for tool_id, entry in self._entries.items()
                if entry.server_name != server_name
            }
            self._rebuild_conflicts()

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._server_generations.clear()
            self._loading = False
            self._refreshing_servers.clear()
            ready_event = self._ready_event
            self._ready_event = None
        if ready_event is not None:
            ready_event.set()

    def snapshot(self) -> tuple[MCPToolEntry, ...]:
        with self._lock:
            return tuple(sorted(self._entries.values(), key=lambda entry: entry.tool_id))

    def get(self, tool_id: str) -> MCPToolEntry | None:
        with self._lock:
            return self._entries.get(tool_id)

    def get_by_model_name(self, model_name: str) -> MCPToolEntry | None:
        matches = [entry for entry in self.snapshot() if entry.model_name == model_name]
        if len(matches) != 1 or matches[0].name_conflict:
            return None
        return matches[0]

    def search(
        self,
        query: str,
        *,
        server_name: str | None = None,
        top_k: int = 5,
    ) -> list[MCPToolEntry]:
        normalized_query = _normalize(query)
        if not normalized_query:
            return []
        ranked: list[tuple[int, str, MCPToolEntry]] = []
        for entry in self.snapshot():
            if server_name and entry.server_name != server_name:
                continue
            normalized_name = _normalize(entry.model_name)
            normalized_id = _normalize(entry.tool_id)
            normalized_server_name = _normalize(
                f"{entry.server_name} {entry.model_name}"
            )
            normalized_desc = _normalize(entry.description)
            if normalized_query == normalized_name:
                score = 0
            elif normalized_query == normalized_id:
                score = 1
            elif normalized_query in normalized_server_name:
                score = 2
            elif len(normalized_name) >= 3 and normalized_name in normalized_query:
                # A model may name several concrete tools in one discovery query.
                # Treat each embedded exact name as a hit instead of requiring
                # every query word to exist in one tool's metadata.
                score = 2
            elif normalized_query in normalized_name:
                score = 3
            elif normalized_query in normalized_desc:
                score = 4
            else:
                words = normalized_query.split()
                haystack = f"{normalized_server_name} {normalized_desc} {normalized_id}"
                if not all(word in haystack for word in words):
                    continue
                score = 5
            ranked.append((score, entry.tool_id, entry))
        ranked.sort(key=lambda item: (item[0], item[1]))
        return [entry for _, _, entry in ranked[: max(1, top_k)]]

    def _rebuild_conflicts(self) -> None:
        counts: dict[str, int] = {}
        for entry in self._entries.values():
            counts[entry.model_name] = counts.get(entry.model_name, 0) + 1
        self._entries = {
            tool_id: MCPToolEntry(
                tool_id=entry.tool_id,
                model_name=entry.model_name,
                server_name=entry.server_name,
                raw_tool_name=entry.raw_tool_name,
                description=entry.description,
                tool=entry.tool,
                generation=entry.generation,
                always_load=entry.always_load,
                name_conflict=counts[entry.model_name] > 1,
            )
            for tool_id, entry in self._entries.items()
        }


_CATALOG = MCPToolCatalog()


def get_mcp_tool_catalog() -> MCPToolCatalog:
    return _CATALOG
