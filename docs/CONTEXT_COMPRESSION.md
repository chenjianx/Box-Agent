# Context Compression

Box-Agent controls context growth at two boundaries:

1. Tool results are persisted before they can dominate a later model request.
2. Conversation history is summarized only when the next request approaches the model's effective input limit.

The two mechanisms are deliberately separate. Tool-result storage is lossless: the complete result remains on disk and the model receives a preview. Context summarization is lossy, so it runs later and preserves a bounded working set.

## Request lifecycle

```text
tool finishes
  -> immediate per-result check
  -> tool messages are appended
  -> before the next LLM request, enforce the fresh-result aggregate budget
  -> estimate the next request size
  -> compact conversation history only when the threshold is reached
  -> call the LLM
```

## Oversized tool-result storage

`ToolResultStorage` in `box_agent/tool_result_storage.py` owns persistence, preview rendering, and per-conversation deduplication. The shared loop invokes it for both sequential and parallel tools; CLI and ACP do not duplicate this policy.

### Immediate per-result check

The default maximum model-facing result is 20,000 characters. A tool can expose `max_result_size_chars`; the effective limit is the smaller of the declaration and the default. Only ordinary results that have not already been processed enter this generic policy.

The following results are not processed a second time:

- a tool result whose `model_context` projection was actually selected is frozen by `tool_use_id`;
- `read_file`, `query_jsonl`, and `search_files` declare infinity and rely on line/character pagination, cursor/structured summarization, and result-count/character pagination respectively;
- `bash` and `bash_output` also declare infinity because they already apply one 50,000-character 40% head + 60% tail truncation inside the tool.

Read-like tools are not externalized by the single-result check because persisting every normal page only to make the model read it again would create a loop. Infinity opts out of that immediate check, but a large parallel batch can still externalize its largest pages to satisfy the aggregate request budget. A tool can also request persistence of complete recoverable text through `ToolResult.persistence_content`.

When an eligible result exceeds the limit:

- a string is saved as `.txt`;
- an array containing only text blocks is serialized as formatted JSON and saved as `.json`;
- files live under `~/.box-agent/sessions/<session>/tool-results/<tool_use_id>.<ext>`;
- exclusive-create (`x`, equivalent to `wx`) prevents overwriting an existing result;
- a stable `<persisted-output>` preview replaces the model-facing content.

The original result is preserved when it is below the limit, contains an image or any other non-text block, or cannot be persisted. Empty output is normalized to `(<Tool name> completed with no output)`; for Bash this is `(Bash completed with no output)`.

Each `tool_use_id` receives one decision. A persisted replacement is cached and reused in later loop iterations, so the result is not written repeatedly. Results already present when a session is resumed are frozen and are not retroactively externalized.

Tools may still bound their own output for operational reasons. When they do, they pass the complete persistable text through `ToolResult.persistence_content`; only `ToolResultStorage` writes it. Bash uses this path for both successful and failed commands: the complete output is saved, while the model keeps the tool-generated head/tail plus the saved path instead of receiving a second generic 2,000-character head preview. Semantic `model_context` projections remain a separate tool concern; once selected, neither the immediate check nor the fresh aggregate budget processes them again.

### Preview format

The preview algorithm:

1. Takes at most the first 2,000 characters.
2. Prefers the last newline within that window when it lies in the second half.
3. Otherwise cuts exactly at 2,000 characters.
4. Adds `...` only when more content exists.

For an ordinary unprocessed result, the model-facing form is:

```text
<persisted-output>
Output too large (...). Full output saved to: ...

Preview (first 2.0KB):
...
</persisted-output>
```

For a self-bounded tool that supplies `persistence_content`, the body is labeled `Tool-bounded output` and contains the tool's existing bounded result instead of another generic preview.

### Aggregate fresh-result budget

Before every LLM request, results first seen during the current conversation are checked against a default 50,000-character aggregate budget. The pass:

1. Considers only fresh `tool_use_id` values.
2. Excludes selected `model_context` projections, but still counts self-bounded tools whose per-result declaration is infinity. Infinity disables immediate per-result persistence; it does not exempt a parallel batch from the aggregate budget.
3. Sorts eligible results from largest to smallest.
4. Uses a path-only persisted wrapper for this aggregate pass and counts the wrapper's actual model-facing length.
5. Persists and replaces the largest results until the actual remaining fresh content is at or below the budget.

Unsupported blocks and persistence failures remain unchanged. Since IDs are marked seen during the pass, later requests do not reconsider the same results. This handles a batch of parallel tool calls whose individual results are below the per-result limit but are too large together.

## Context-limit compaction

### Threshold

The trigger is derived from the model's input budget:

```text
autoCompactThreshold = 0.9 * (context_window - max_output_tokens)
```

`LLMConfig.context_token_limit` reserves the configured maximum output budget,
then keeps 10% of the remaining input budget as headroom for estimation drift
and the summary request.

### Estimating the next request

Provider clients attach usage metadata to the assistant message produced by a real API response. Compaction finds the newest such message and computes its complete response context as:

```text
input_tokens
+ cache_creation_input_tokens
+ cache_read_input_tokens
+ output_tokens
```

It then estimates messages appended after that response conservatively. If no message has real API usage, the entire pending request—including tool schemas—is estimated with the larger of `characters / 4` and UTF-8 bytes `/ 3`. This avoids severe under-counting for CJK and other multibyte text.

### Compacted message layout

When a recoverable workflow provides a current filesystem-derived checkpoint that fits the bounded checkpoint budget, compaction rebuilds directly from that checkpoint, the exact latest user message, the protocol-complete bounded recent group, and runtime state. This `checkpoint` mode performs no summary-model call. Other turns make one summary request by appending a temporary `user` instruction to the exact existing message list. It does not serialize messages into a new prompt and does not split or roll the source. This preserves the complete provider message prefix so the summary request can reuse its KV cache. ACP resolves this call through the session model router and caps it at 4,096 output tokens. Automatic sessions may select only from the host-provided `autoRouting.models` pool, ranked by the summary task tags, ability, and context fit. Explicitly selected sessions stay on that exact model; there is no independent lite-model switch. Other hosts may supply a dedicated summary client or fall back to the main client. Tools and thinking are disabled. The instruction requires a chronological list of every user message and places all structured analysis inside one `<summary>...</summary>` block, with an embedded nine-section output example. The response must consist of exactly one non-empty summary block; only its inner text is written after `Summary:` and the tags are discarded. The requested and locally enforced summary limit is 8,000 characters; if the rebuilt request still exceeds the safe input limit, the same summary is deterministically tightened to 4,000 and finally 2,000 characters without another provider call. If the call fails, is malformed, or returns empty output, an explicitly lossy deterministic bounded fallback is used.

The model output is wrapped in this synthetic `user` message:

```text
This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.

Summary:
<model-generated summary>

Continue the conversation from where it left off without asking the user any further questions. Resume directly — do not acknowledge the summary, do not recap what was happening, do not preface with "I'll continue" or similar. Pick up the last task as if the break never happened.
```

The rebuilt history is ordered as follows:

```text
system message
summary user message
bounded recent messages
runtime-state user message
```

Recent selection applies to user, assistant, and tool messages. Assistant tool calls stay grouped with their contiguous tool results. Selection targets at most 5 messages and 20,000 total characters, and the latest real user message is always retained verbatim. When the newest complete protocol group alone exceeds the character cap, compaction keeps the exact assistant tool calls, arguments, result IDs, ordering, and result count while replacing already-summarized assistant reasoning and tool-result bodies with bounded receipts. A request is blocked only when the remaining exact user text, tool-call arguments, system instructions, tool schemas, bounded summary, and runtime state still cannot fit.

Compaction does not discover, reread, or replay recent files.

Current goal, todo, and plan state are read through their explicit, side-effect-free `compaction_state` contract; compaction never invokes a normal tool call. Their combined runtime-state message is capped at 12,000 characters. Full active skill instructions remain pinned in the system message and are not reconstructed by replaying historical `get_skill` calls. Internal summary/runtime-state messages are excluded whenever control policy asks for the latest real user text.

If the rebuilt request still exceeds the safe limit, the outcome is marked blocked instead of silently sending a known-oversized request.

## Adjacent protections

Write/edit tool-call arguments remain verbatim until whole-history compaction summarizes their turn; they are not independently replaced with history placeholders. This is separate from tool-result storage. A legacy safety guard still prevents placeholders from older or externally supplied sessions from being reused as executable file or code arguments.

## Verification

Direct regression coverage lives in:

- `tests/test_tool_result_storage.py` for type handling, exclusive writes, previews, Read single-result opt-out, failures, deduplication, and aggregate ordering;
- `tests/test_core.py` for pre-request enforcement, usage-plus-delta estimation, exact-prefix one-shot summarization, fallback estimation, bounded retention, and runtime-state restoration;
- `tests/test_auth.py` for the derived threshold.
