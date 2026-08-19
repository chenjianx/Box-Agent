"""Turn-scoped ownership for the shared Playwright MCP browser.

OfficeV3 exposes one Playwright MCP stdio session to an ACP process. Multiple
local-agent sessions may therefore reach the same browser context concurrently.
This coordinator gives the first Playwright call in a turn an exclusive lease
until that turn ends, while allowing parallel Playwright calls from the same
turn to proceed.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar, Token

_current_owner: ContextVar[str | None] = ContextVar(
    "box_agent_browser_runtime_owner", default=None
)


class BrowserRuntimeCoordinator:
    """Serialize distinct ACP turns that share one Playwright MCP session."""

    _condition: asyncio.Condition | None = None
    _loop: asyncio.AbstractEventLoop | None = None
    _owner: str | None = None

    @classmethod
    def _condition_for_current_loop(cls) -> asyncio.Condition:
        loop = asyncio.get_running_loop()
        if cls._condition is None or cls._loop is not loop:
            if cls._owner is not None:
                raise RuntimeError("cannot move an active browser lease between event loops")
            cls._condition = asyncio.Condition()
            cls._loop = loop
        return cls._condition

    @classmethod
    async def acquire(cls, owner: str) -> None:
        condition = cls._condition_for_current_loop()
        async with condition:
            await condition.wait_for(lambda: cls._owner in (None, owner))
            cls._owner = owner

    @classmethod
    async def release(cls, owner: str) -> None:
        condition = cls._condition_for_current_loop()
        async with condition:
            if cls._owner != owner:
                return
            cls._owner = None
            condition.notify_all()


def set_browser_runtime_owner(owner: str) -> Token[str | None]:
    return _current_owner.set(owner)


def reset_browser_runtime_owner(token: Token[str | None]) -> None:
    _current_owner.reset(token)


async def acquire_browser_runtime_for_current_turn() -> None:
    owner = _current_owner.get()
    if owner is not None:
        await BrowserRuntimeCoordinator.acquire(owner)


async def release_browser_runtime_for_current_turn() -> None:
    owner = _current_owner.get()
    if owner is not None:
        await BrowserRuntimeCoordinator.release(owner)


async def release_browser_runtime(owner: str) -> None:
    await BrowserRuntimeCoordinator.release(owner)
