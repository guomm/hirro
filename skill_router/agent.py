from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from skill_router.executor import Executor
from skill_router.llm import LLMClient
from skill_router.mcp import McpToolRegistry
from skill_router.models import Skill, SkillRouterError, ToolSpec
from skill_router.tools import ToolRegistry


@dataclass(frozen=True)
class AgentResult:
    content: str
    messages: list[dict[str, str]] = field(default_factory=list)


class AgentRunner:
    def __init__(
        self,
        client: LLMClient,
        skills: list[Skill],
        builtin_tools: ToolRegistry,
        mcp_tools: McpToolRegistry,
        executor: Executor,
        max_steps: int = 10,
        verbose: bool = False,
    ):
        self.client = client
        self.skills = {skill.name: skill for skill in skills}
        self.builtin_tools = builtin_tools
        self.mcp_tools = mcp_tools
        self.executor = executor
        self.max_steps = max_steps
        self.verbose = verbose
        self._activated_skills: set[str] = set()

    def run(
        self,
        user_input: str,
        history: Optional[list[dict[str, str]]] = None,
        dry_run: bool = False,
    ) -> AgentResult:
        turn_messages: list[dict[str, str]] = [{"role": "user", "content": user_input}]
        messages = [
            {"role": "system", "content": self._system_prompt()},
            *(history or []),
            *turn_messages,
        ]

        for _ in range(self.max_steps):
            if self.verbose:
                print("LLM request messages:")
                print(json.dumps(messages, ensure_ascii=False, indent=2))
            envelope = self.client.complete_json(messages)
            if self.verbose:
                print("LLM response:")
                print(json.dumps(envelope, ensure_ascii=False, indent=2))
            envelope_type = envelope.get("type")
            if envelope_type == "final":
                content = _require_string(envelope, "content")
                turn_messages.append({"role": "assistant", "content": content})
                return AgentResult(content=content, messages=turn_messages)

            if envelope_type != "function_call":
                raise SkillRouterError(
                    "Agent response must be a final answer or function_call"
                )

            name = _require_string(envelope, "name")
            arguments = envelope.get("arguments", {})
            if not isinstance(arguments, dict):
                raise SkillRouterError("Function call arguments must be an object")

            call_text = json.dumps(envelope, ensure_ascii=False)
            turn_messages.append({"role": "assistant", "content": call_text})
            messages.append({"role": "assistant", "content": call_text})

            if dry_run:
                content = f"Planned function call: {call_text}"
                turn_messages.append({"role": "assistant", "content": content})
                return AgentResult(content=content, messages=turn_messages)

            try:
                result = self._call_function(name, arguments)
            except SkillRouterError as exc:
                result = {
                    "ok": False,
                    "error": str(exc),
                    "hint": (
                        "Fix the function arguments or choose another function. "
                        "You may call list_capabilities or activate_skill to inspect valid names."
                    ),
                }
            result_text = json.dumps(
                {"type": "function_result", "name": name, "result": result},
                ensure_ascii=False,
            )
            turn_messages.append({"role": "user", "content": result_text})
            messages.append({"role": "user", "content": result_text})

        raise SkillRouterError(f"Agent exceeded max_steps={self.max_steps}")

    def _call_function(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "list_capabilities":
            return self._list_capabilities()
        if name == "activate_skill":
            return self._activate_skill(arguments)
        if name == "list_tools":
            return self._list_tools(arguments)
        if name == "call_tool":
            return self._call_tool(arguments)
        if name == "run_skill_command":
            return self._run_skill_command(arguments)
        if name == "run_skill_script":
            return self._run_skill_script(arguments)
        raise SkillRouterError(f"Unknown function call: {name}")

    def _list_capabilities(self) -> dict[str, Any]:
        return {
            "skills": [
                {"name": skill.name, "description": skill.description}
                for skill in self.skills.values()
            ],
            "builtin_tools": [
                _tool_to_dict(tool) for tool in self.builtin_tools.specs()
            ],
            "mcp_servers": self.mcp_tools.server_summaries(),
        }

    def _list_tools(self, arguments: dict[str, Any]) -> dict[str, Any]:
        server_name = _optional_string(arguments, "server")
        if server_name is None:
            return {
                "builtin_tools": [
                    _tool_to_dict(tool) for tool in self.builtin_tools.specs()
                ],
                "mcp_servers": self.mcp_tools.server_summaries(),
            }
        return {
            "server": server_name,
            "tools": [
                _tool_to_dict(tool)
                for tool in self.mcp_tools.specs_for_server(server_name)
            ],
        }

    def _activate_skill(self, arguments: dict[str, Any]) -> dict[str, Any]:
        skill_name = _require_string(arguments, "name")
        skill = self.skills.get(skill_name)
        if skill is None:
            raise SkillRouterError(f"Unknown skill: {skill_name}")
        self._activated_skills.add(skill.name)
        referenced_tools = _extract_qualified_tool_refs(skill.body)
        suggested_servers = self._suggest_mcp_servers(skill)
        return {
            "name": skill.name,
            "description": skill.description,
            "instructions": skill.body,
            "referenced_tools": [
                {"server": server, "name": tool}
                for server, tool in sorted(referenced_tools)
            ],
            "suggested_mcp_servers": suggested_servers,
            "tool_discovery": [
                {"function": "list_tools", "arguments": {"server": server}}
                for server in suggested_servers
            ],
            "commands": [
                {
                    "id": command.id,
                    "description": command.description,
                    "args": command.args,
                    "run": command.run,
                }
                for command in skill.commands.values()
            ],
            "script_execution": {
                "function": "run_skill_script",
                "description": (
                    "Use this only when the skill instructions explicitly require running a "
                    "local Python script from the skill directory. Pass a relative script_path "
                    "and argv-style args."
                ),
            },
        }

    def _run_skill_command(self, arguments: dict[str, Any]) -> dict[str, Any]:
        skill_name = _require_string(arguments, "skill")
        command_id = _require_string(arguments, "name")
        command_args = arguments.get("args", {})
        if not isinstance(command_args, dict):
            raise SkillRouterError("run_skill_command.args must be an object")
        if skill_name not in self._activated_skills:
            raise SkillRouterError(
                f"Skill must be activated before execution: {skill_name}"
            )
        skill = self.skills.get(skill_name)
        if skill is None:
            raise SkillRouterError(f"Unknown skill: {skill_name}")
        result = self.executor.execute(skill, command_id, command_args)
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def _run_skill_script(self, arguments: dict[str, Any]) -> dict[str, Any]:
        skill_name = _require_string(arguments, "skill")
        script_path = _require_string(arguments, "name")
        script_args = arguments.get("args", [])
        if not isinstance(script_args, list) or not all(
            isinstance(arg, str) for arg in script_args
        ):
            raise SkillRouterError("run_skill_script.args must be a string list")
        if skill_name not in self._activated_skills:
            raise SkillRouterError(
                f"Skill must be activated before script execution: {skill_name}"
            )
        skill = self.skills.get(skill_name)
        if skill is None:
            raise SkillRouterError(f"Unknown skill: {skill_name}")
        result = self.executor.execute_script(skill, script_path, script_args)
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def _call_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        tool_name = _require_string(arguments, "name")
        tool_args = arguments.get("args", {})
        if not isinstance(tool_args, dict):
            raise SkillRouterError("call_tool.args must be an object")
        qualified = _split_qualified_tool_name(tool_name)
        if qualified is not None:
            server_name, unqualified_tool_name = qualified
            self._require_loaded_mcp_server(server_name)
            return self.mcp_tools.execute(server_name, unqualified_tool_name, tool_args)
        if self._find_builtin_tool(tool_name) is not None:
            return self.builtin_tools.execute(tool_name, tool_args)
        server_name = _optional_string(arguments, "server")
        if server_name is None:
            raise SkillRouterError(
                f"server is required for MCP tool {tool_name!r}; "
                "use name='server.tool' or pass server explicitly"
            )
        self._require_loaded_mcp_server(server_name)
        return self.mcp_tools.execute(server_name, tool_name, tool_args)

    def _find_builtin_tool(self, name: str) -> Optional[ToolSpec]:
        matches = [tool for tool in self.builtin_tools.specs() if tool.name == name]
        return matches[0] if len(matches) == 1 else None

    def _suggest_mcp_servers(self, skill: Skill) -> list[str]:
        known_servers = {server["name"]: server for server in self.mcp_tools.server_summaries()}
        referenced = _extract_qualified_tool_refs(skill.body)
        referenced_servers = [
            server for server in sorted({server for server, _ in referenced})
            if server in known_servers
        ]
        if referenced_servers:
            return referenced_servers

        haystack = f"{skill.name}\n{skill.description}\n{skill.body}".lower()
        skill_tool_names = _extract_function_like_names(haystack)
        inferred: list[str] = []
        for server in known_servers.values():
            server_text = "\n".join(
                str(server.get(key, ""))
                for key in ("name", "type", "note")
            ).lower()
            server_tool_names = _extract_function_like_names(server_text)
            if skill_tool_names & server_tool_names:
                inferred.append(server["name"])
            elif server["name"].lower() in haystack or any(
                token and token in haystack
                for token in server_text.replace("-", " ").replace("_", " ").split()
                if len(token) >= 5
            ):
                inferred.append(server["name"])
        return inferred

    def _require_loaded_mcp_server(self, server_name: str) -> None:
        if not self.mcp_tools.has_loaded_server(server_name):
            raise SkillRouterError(
                f"MCP tools for server {server_name!r} are not loaded. "
                f"Call list_tools with {{\"server\":\"{server_name}\"}} first, then call_tool using the returned schema."
            )

    def _system_prompt(self) -> str:
        capabilities = self._list_capabilities()
        return (
            "You are a local tool-using agent. Return exactly one JSON object per turn.\n"
            'Final answer: {"type":"final","content":"..."}\n'
            'Function call: {"type":"function_call","name":"FUNCTION","arguments":{...}}\n'
            "Allowed functions and argument schemas:\n"
            "- list_capabilities: {}\n"
            '- list_tools: {"server":"optional-mcp-server"}\n'
            '- activate_skill: {"name":"skill-name"}\n'
            '- call_tool: {"name":"builtin-tool OR mcp-tool OR server.tool","args":{...},"server":"optional-mcp-server"}\n'
            '- run_skill_command: {"skill":"skill-name","name":"command-id","args":{...}}\n'
            '- run_skill_script: {"skill":"skill-name","name":"relative/script.py","args":["argv"]}\n'
            "Rules: use only these function names; use name/args/server consistently; "
            "when calling an MCP tool, use name='server.tool' or pass server explicitly; "
            "before calling any MCP tool, call list_tools for its server and use the returned schema; "
            "after activating a skill, prefer its suggested_mcp_servers for list_tools; "
            "activate a skill before running its command or script; follow activated skill "
            "Markdown as workflow instructions; do not invent capability names; on function "
            "errors, fix the next call or explain the error.\n"
            "Capabilities: " + json.dumps(capabilities, ensure_ascii=False)
        )


def _tool_to_dict(tool: ToolSpec) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "args": tool.args,
        "source": tool.source,
        "server": tool.server,
    }


def _require_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SkillRouterError(f"Expected non-empty string field: {key}")
    return value.strip()


def _optional_string(data: dict[str, Any], key: str) -> Optional[str]:
    value = data.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _extract_function_like_names(text: str) -> set[str]:
    return {
        match.group(1).lower()
        for match in re.finditer(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", text)
    }


def _extract_qualified_tool_refs(text: str) -> set[tuple[str, str]]:
    return {
        (match.group(1), match.group(2))
        for match in re.finditer(
            r"\b([A-Za-z0-9_-]+)\.([A-Za-z_][A-Za-z0-9_]*)\s*\(",
            text,
        )
    }


def _split_qualified_tool_name(name: str) -> Optional[tuple[str, str]]:
    if "." not in name:
        return None
    server, tool = name.split(".", 1)
    if not server.strip() or not tool.strip():
        return None
    return server.strip(), tool.strip()
