# Context Compression Design

## Background

`skill-router` runs an internal multi-step agent loop in chat mode. A single user
turn may include:

- skill activation
- MCP `list_tools`
- one or more MCP `call_tool`
- local skill command or script execution
- final natural-language answer

The original implementation appended the full turn transcript back into chat
history. That was simple, but it created a fast-growing prompt.

## Problems

### 1. Tool results were too large

MCP tools often return large JSON payloads:

- SQL result sets
- validation reports
- long `content` arrays
- nested `structuredContent`

If these results are kept in raw form, later turns keep re-sending them to the
model.

### 2. Tool schemas were repeated or lost

For MCP tools, the model needs schema information such as:

- server name
- tool name
- argument names and simple types

If tool schemas stay only inside old turn history, aggressive history compression
can remove them. Then the model may still know that a tool exists, but no longer
remember its argument shape.

### 3. Message-level truncation lost important state

A naive approach is to keep only the latest N messages. That breaks down because
the important information is distributed across a turn:

- the user goal
- the tool sequence
- the final answer

Dropping messages mechanically can keep the wrong half of a turn and discard the
actual useful part.

### 4. Static prompt and runtime state were mixed together

Originally, capability data and runtime state tended to drift toward the system
prompt. That makes the system prompt larger and blurs the boundary between:

- stable protocol rules
- dynamic session state

## Design Goals

The compression design was built around four goals:

1. Keep the system prompt stable and small.
2. Preserve enough runtime state for the next turn to continue correctly.
3. Remove large raw tool outputs from future prompts.
4. Avoid depending on an extra LLM summarization call for every turn.

## Solution

### 1. Turn-level compression instead of raw transcript replay

Instead of appending the full internal agent trace to history, each completed
turn is reduced to:

- the user message
- a compact `Turn summary`
- the final assistant answer

This preserves the user intent, the executed steps, and the user-facing result,
without replaying every internal `function_call` and full `function_result`.

Relevant code:

- [context.py](d:/git_task/hirro/skill_router/context.py)
- [agent.py](d:/git_task/hirro/skill_router/agent.py)

### 2. Large results are stored as artifacts

When a tool result grows beyond a configured threshold, it is not kept inline in
full. Instead:

- the raw payload is written to `.skill-router/artifacts/`
- the agent keeps only:
  - `summary`
  - `preview`
  - `artifact_id`

This removes the large payload from future turns while still making it
recoverable.

Relevant code:

- `ArtifactStore` in [context.py](d:/git_task/hirro/skill_router/context.py)
- `read_artifact` built-in tool in [tools.py](d:/git_task/hirro/skill_router/tools.py)

### 3. Generic rule-based payload summarization

Tool result formats differ across skills, MCP servers, and tools, so the system
does not rely on `stdout`/`stderr` only. The current summarizer extracts useful
signals from generic JSON structures:

- scalar fields such as `ok`, `status`, `decision`, `message`, `error`
- nested objects such as `result`, `meta`, `structuredContent`
- lists summarized as size plus column-like field names

Example summary shape:

```text
content=list[1]{type, text}; structuredContent{status=prechecked; decision=can_execute_via_tool; sql=select ...}
```

This is cheap, deterministic, and works well enough for most MCP JSON payloads.

Relevant code:

- `summarize_payload`
- `summarize_dict`
- `build_preview`

All in [context.py](d:/git_task/hirro/skill_router/context.py).

### 4. Conversation memory compacts by turn

`ConversationMemory` no longer truncates raw messages. It stores recent turns
and compacts old turns into a structured conversation summary:

- user input
- turn steps
- final answer

That means compaction preserves the meaningful unit of interaction instead of
cutting across tool calls arbitrarily.

Relevant code:

- `ConversationMemory`
- `summarize_turn`

In [context.py](d:/git_task/hirro/skill_router/context.py).

### 5. LLM session summarization after 10 turns

Rule-based compression is used for every turn, but session-level compression is
not limited to rules only.

The current chat memory policy is:

- every turn is reduced to `user + Turn summary + final answer`
- recent raw turns are kept in memory
- once the session grows beyond `10` turns, old turns are compacted

When turn compaction is triggered, the system first tries to summarize the old
turn block with the same LLM client used by the router. The prompt asks for a
compact factual continuation summary that preserves:

- user goals
- completed actions
- important results
- active skills
- loaded servers or tools when relevant
- pending follow-ups

If the LLM summary fails for any reason, the system falls back automatically to
the existing rule-based turn summarizer. Chat continues normally even if the
LLM-based summary path fails.

This gives the design two levels of compression:

- rule-based per-turn compression for every turn
- LLM-based session compression only when the conversation is long enough

Relevant code:

- `ConversationMemory`
- `_summarize_compacted_turns`

In [context.py](d:/git_task/hirro/skill_router/context.py), and the chat
summarizer wiring in [cli.py](d:/git_task/hirro/skill_router/cli.py).

### 6. Runtime state is split out from the system prompt

The system prompt now contains only static protocol rules:

- valid internal function names
- required argument shapes
- skill and MCP calling rules

Dynamic runtime state is emitted as a separate runtime context message, which
includes:

- `skills`
- `builtin_tools`
- `mcp_servers`
- `activated_skills`
- `loaded_servers`
- `loaded_tools`

This keeps the system prompt stable while still letting the model see the state
it needs for the current session.

Relevant code:

- `_system_prompt`
- `_runtime_context_message`
- `_list_capabilities`

In [agent.py](d:/git_task/hirro/skill_router/agent.py).

### 7. Loaded MCP tool schemas are retained in minimal form

After a server has been listed once, the system remembers the minimal schema of
its loaded tools:

- `server`
- `name`
- `args`

This solves a practical problem in multi-turn chat:

- execution already knows a server is loaded
- but the model may forget the tool argument schema after history compression

By keeping `loaded_tools` in runtime context, later turns can continue to call
previously loaded MCP tools without forcing another full `list_tools`.

Relevant code:

- `_loaded_tool_summaries` in [agent.py](d:/git_task/hirro/skill_router/agent.py)
- `loaded_server_names` in [mcp.py](d:/git_task/hirro/skill_router/mcp.py)

## Effects

### 1. Prompt growth is substantially reduced

The original design repeatedly replayed:

- tool schemas
- raw MCP results
- full internal function traces

The current design keeps only compact turn records plus minimal runtime state.
In MCP-heavy workflows such as SQL precheck and execution, this usually reduces
prompt-history volume by a large margin.

In practice, the expected reduction is roughly:

- moderate tool usage: `70%` to `80%`
- MCP / SQL heavy usage: `80%` to `90%+`

The exact number depends on how large raw tool payloads are.

### 2. Multi-turn continuity is more stable

The model now keeps access to:

- activated skills
- loaded MCP servers
- loaded tool schemas

even after old turns are compacted.

### 3. Long sessions now have a higher-quality summary path

Short and medium conversations still rely on cheap rule-based summaries, but
long sessions now gain an extra LLM summarization pass after the turn threshold
is exceeded. This improves retention of:

- cross-turn intent
- important results
- pending tasks
- state that is semantically important but not obvious from field names alone

### 4. Failure handling remains controlled

The project keeps a strict function-call protocol. Compression does not relax
that protocol. Invalid `function_call` envelopes are repaired through the
existing error-feedback loop rather than silently inventing new calling formats.

## Tradeoffs

### 1. Rule-based summaries are approximate

The generic summarizer is intentionally simple and cheap. It works well for many
JSON payloads, but it cannot extract every domain-specific detail perfectly.

For example:

- Impala SQL results
- anti-spam operation reports
- governance configuration results

may still benefit from tool-specific summarizers later.

### 2. Runtime context still consumes tokens

Moving `loaded_tools` and session state out of the system prompt improves the
design boundary, but it does not make those tokens free. If many servers and
many tools are loaded in one session, runtime context can still grow.

### 3. LLM session summaries add cost when the threshold is crossed

The LLM-based session summary is not called on every turn, but it is still an
extra model call when compaction is triggered. This is a deliberate tradeoff:

- pay a small extra cost occasionally
- reduce repeated prompt replay across many later turns

### 4. Artifacts require explicit recovery

Once a large payload is stored as an artifact, the model only sees the summary
and preview by default. If it needs the full content, it must use
`read_artifact`.

## Observability

In `--verbose` mode, chat now prints context-compression events when old turns
are compacted. The log shows:

- how many old turns were compacted
- whether the system used `llm` or `rule` summarization
- how many recent turns were kept

Example:

```text
context compression triggered: compacted 8 old turns using llm summary; kept 3 recent turns
```

## Current Result

The current context-compression design provides:

- turn-level history compression
- artifact-backed large result storage
- generic JSON result summarization
- turn-based conversation compaction
- LLM-based session summary after the turn threshold
- runtime-state separation from static system rules
- retained minimal schemas for loaded MCP tools

This is enough to make multi-turn MCP-heavy workflows cheaper and more stable
without introducing an extra LLM summarization step on every turn.

## Future Work

The next improvements, if needed, are:

1. Add per-tool or per-server custom summarizers for high-value MCP tools.
2. Add token estimation and compression metrics in debug mode.
3. Optionally introduce an LLM summarizer only when rule-based summaries are not
   sufficient or when history crosses a larger threshold.
