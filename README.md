# Skill Router

Route a natural-language task to one of three local capability types with an
OpenAI-compatible model:

- built-in Python tools
- MCP tools from configured stdio or HTTP MCP servers
- local skills with whitelisted command execution

## Requirements

- Python 3.12
- `OPENAI_API_KEY`
- An OpenAI-compatible `/chat/completions` endpoint
- Optional MCP stdio or HTTP servers

## Run

Single turn:

```powershell
py -3.12 -m skill_router "echo hello" --skills-dir ./skills --model gpt-4.1-mini
```

Interactive multi-turn chat:

```powershell
py -3.12 -m skill_router --chat --skills-dir ./skills --model gpt-4.1-mini
```

If you omit the task, the CLI also starts chat mode:

```powershell
py -3.12 -m skill_router --skills-dir ./skills --model gpt-4.1-mini
```

Use a custom compatible endpoint:

```powershell
py -3.12 -m skill_router "echo hello" --skills-dir ./skills --model your-model --base-url https://example.com/v1
```

Dry-run the first model function call without executing:

```powershell
py -3.12 -m skill_router "echo hello" --skills-dir ./skills --model your-model --plan-only
```

In chat mode, each user message runs an internal agent loop. The model may call
one or more functions, receive their results, and then return a final
plain-language answer. The turn transcript is appended to in-memory history so
follow-up requests can refer to earlier messages or results. Type `/exit` or
`/quit` to stop.

Use MCP servers from a custom config path:

```powershell
py -3.12 -m skill_router "search my docs" --model your-model --mcp-config .skill-router/mcp.json
```

## Built-In Tools

The router ships with a few safe built-in tools:

- `calculator`: evaluates basic arithmetic expressions.
- `current_time`: returns current time for an IANA timezone.
- `text_stats`: counts characters, words, and lines.

Built-in tools run in-process and never invoke a shell.

## MCP Tools

MCP servers are configured in `.mcp.json` by default. The router supports the
common `mcpServers` shape:

```json
{
  "mcpServers": {
    "impala-mcp": {
      "type": "http",
      "url": "http://bi-metabase.vvic.com:8686/mcp/impala-query/",
      "note": "Impala MCP service"
    }
  }
}
```

It also supports the older local `servers` shape for stdio servers:

```json
{
  "servers": {
    "demo": {
      "type": "stdio",
      "command": "py",
      "args": ["-3.12", "path/to/mcp_server.py"],
      "env": {
        "TOKEN": "optional"
      }
    }
  }
}
```

At startup, the router calls `tools/list` on each configured MCP server and
includes only MCP server summaries in the model's capability catalog. Tool
details are loaded on demand with `list_tools`. If the model selects an MCP
tool, the router calls `tools/call` with the model-provided arguments. HTTP
servers are called with JSON-RPC POST requests and support JSON or SSE JSON
responses.

## Skill Manifest

Each skill is a directory containing `SKILL.md`. The executable command is
declared in frontmatter:

```yaml
---
name: echo
description: Echo text for smoke testing the skill router.
command:
  id: echo_text
  run: py -3.12 scripts/echo.py
  args:
    text: string
  description: Print the supplied text.
---
```

The model can select a skill and fill arguments, but it cannot rewrite
`command.run`. Execution uses `subprocess.run(..., shell=False)`.

## Agent Response Shape

The model always returns one JSON object. To call a function:

```json
{
  "type": "function_call",
  "name": "call_tool",
  "arguments": {
    "name": "calculator",
    "args": {"expression": "2 + 2"}
  }
}
```

After the program executes the function, the result is sent back to the model.
The model can call another function or return a final answer:

```json
{
  "type": "final",
  "content": "2 + 2 = 4."
}
```

Available internal functions:

- `list_capabilities`
- `list_tools`
- `activate_skill`
- `call_tool`
- `run_skill_command`
- `run_skill_script`

The model must call `activate_skill` before `run_skill_command` or
`run_skill_script`. The activated skill Markdown body is the primary workflow
instruction source. Frontmatter `command` is optional and only acts as a
structured shortcut. If a skill describes several commands in its body, the
model should follow the body and call MCP tools or `run_skill_script` as needed.

Function arguments use one consistent naming scheme:

- `name`: selected skill, tool, command, or script path.
- `args`: inputs for that capability; object for tools/commands, string list for scripts.
- `server`: optional MCP server name when selecting a tool from a specific server.
- `skill`: skill name for `run_skill_command` and `run_skill_script`.

Use `list_tools` to dynamically load MCP tool details without putting every MCP
tool in the initial system prompt:

```json
{
  "type": "function_call",
  "name": "list_tools",
  "arguments": {
    "server": "impala-mcp"
  }
}
```

MCP example:

```json
{
  "type": "function_call",
  "name": "call_tool",
  "arguments": {
    "server": "impala-mcp",
    "name": "precheck_sql",
    "args": {"sql": "select 1"}
  }
}
```

`run_skill_script` only runs relative `.py` files inside the activated skill
directory, uses `shell=False`, and goes through whitelist confirmation.

## Whitelist

Skill commands are whitelisted per project in `.skill-router/whitelist.json`
using:

- skill name
- command id
- fixed run template

The first execution asks for confirmation. If accepted, the command is added to
the whitelist and later runs do not prompt again for the same command template.
