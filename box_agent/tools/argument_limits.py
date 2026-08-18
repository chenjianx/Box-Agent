"""Central safety limits for model-generated tool arguments.

These are runtime safety invariants rather than user-tunable product budgets.
The streamed JSON limits intentionally leave room for escaping while stopping
pathological calls before they consume an entire provider completion budget.
"""

from __future__ import annotations

MAX_GENERATED_BODY_CHARS = 12_000
RECOMMENDED_GENERATED_BODY_CHARS = 5_500
MAX_BASH_COMMAND_CHARS = 8_000

_STREAM_ARGUMENT_LIMITS: dict[str, int | None] = {
    "bash": 10_000,
    # write_file is intentionally bounded only by the provider completion
    # budget. A complete call is safe to execute regardless of content size;
    # provider-truncated calls are discarded by the agent loop.
    "write_file": None,
    "append_file": 16_000,
    "execute_code": 16_000,
    "staged_file_write": 16_000,
    "edit_file": 24_000,
}
DEFAULT_STREAM_ARGUMENT_CHARS = 24_000
TOOL_ARGUMENT_ACTIVITY_BUCKET_CHARS = 2_048
# Provider clients aggregate raw tool-call deltas before the agent loop sees
# them. Emit a bounded liveness event while that aggregation is progressing so
# the outer stale detector observes real provider traffic instead of mistaking
# a slowly generated JSON argument for a dead stream.
PROVIDER_STREAM_ACTIVITY_INTERVAL_SECONDS = 5.0


def streamed_argument_limit(tool_name: str | None) -> int | None:
    """Return the raw streamed-JSON budget, or ``None`` for no local cap."""
    return _STREAM_ARGUMENT_LIMITS.get(tool_name or "", DEFAULT_STREAM_ARGUMENT_CHARS)
