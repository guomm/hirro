from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from skill_router.models import SkillRouterError, ToolSpec


@dataclass(frozen=True)
class McpServerConfig:
    name: str
    type: str = "stdio"
    command: Optional[str] = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    url: Optional[str] = None
    note: str = ""


class McpConfig:
    def __init__(self, servers: list[McpServerConfig]):
        self.servers = {server.name: server for server in servers}

    @classmethod
    def load_optional(cls, path: Path | str) -> "McpConfig":
        config_path = Path(path)
        if not config_path.exists():
            return cls([])
        data = json.loads(config_path.read_text(encoding="utf-8"))
        raw_servers = data.get("mcpServers", data.get("servers", {}))
        if not isinstance(raw_servers, dict):
            raise SkillRouterError("MCP config field 'mcpServers' or 'servers' must be an object")

        servers: list[McpServerConfig] = []
        for name, raw in raw_servers.items():
            if not isinstance(name, str) or not isinstance(raw, dict):
                raise SkillRouterError("Each MCP server entry must be an object")
            server_type = raw.get("type", "stdio")
            if not isinstance(server_type, str):
                raise SkillRouterError(f"MCP server {name!r} type must be a string")
            server_type = server_type.lower()
            args = raw.get("args", [])
            env = raw.get("env", {})
            headers = raw.get("headers", {})
            if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
                raise SkillRouterError(f"MCP server {name!r} args must be a string list")
            if not isinstance(env, dict) or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in env.items()
            ):
                raise SkillRouterError(f"MCP server {name!r} env must map strings to strings")
            if not isinstance(headers, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in headers.items()
            ):
                raise SkillRouterError(f"MCP server {name!r} headers must map strings to strings")
            if server_type == "stdio":
                command = raw.get("command")
                if not isinstance(command, str) or not command:
                    raise SkillRouterError(f"MCP stdio server {name!r} must declare command")
                servers.append(
                    McpServerConfig(
                        name=name,
                        type="stdio",
                        command=command,
                        args=args,
                        env=env,
                        headers=headers,
                        note=str(raw.get("note", "")),
                    )
                )
            elif server_type == "http":
                url = raw.get("url")
                if not isinstance(url, str) or not url:
                    raise SkillRouterError(f"MCP HTTP server {name!r} must declare url")
                servers.append(
                    McpServerConfig(
                        name=name,
                        type="http",
                        url=normalize_http_url(url),
                        env=env,
                        headers=headers,
                        note=str(raw.get("note", "")),
                    )
                )
            else:
                raise SkillRouterError(
                    f"MCP server {name!r} has unsupported type {server_type!r}"
                )
        return cls(servers)


class McpToolRegistry:
    def __init__(self, config: McpConfig):
        self.config = config
        self._tools: dict[tuple[str, str], ToolSpec] = {}
        self._loaded_servers: set[str] = set()

    def server_summaries(self) -> list[dict[str, str]]:
        return [
            {"name": server.name, "type": server.type, "note": server.note}
            for server in self.config.servers.values()
        ]

    def discover(self, server_name: Optional[str] = None) -> list[ToolSpec]:
        servers = (
            [self.config.servers[server_name]]
            if server_name is not None and server_name in self.config.servers
            else list(self.config.servers.values())
            if server_name is None
            else []
        )
        if server_name is not None and not servers:
            raise SkillRouterError(f"Unknown MCP server: {server_name}")
        if server_name is None:
            self._tools = {}
            self._loaded_servers = set()
        for server in servers:
            with create_session(server) as session:
                for tool in session.list_tools():
                    self._tools[(server.name, tool.name)] = tool
            self._loaded_servers.add(server.name)
        if server_name is None:
            return list(self._tools.values())
        return [
            tool
            for (tool_server, _), tool in self._tools.items()
            if tool_server == server_name
        ]

    def specs(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def specs_for_server(self, server_name: str) -> list[ToolSpec]:
        if server_name in self._loaded_servers:
            return [
                tool
                for (tool_server, _), tool in self._tools.items()
                if tool_server == server_name
            ]
        return self.discover(server_name)

    def has_loaded_server(self, server_name: str) -> bool:
        return server_name in self._loaded_servers

    def loaded_server_names(self) -> list[str]:
        return sorted(self._loaded_servers)

    def execute(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        server = self.config.servers.get(server_name)
        if server is None:
            raise SkillRouterError(f"Unknown MCP server: {server_name}")
        with create_session(server) as session:
            return session.call_tool(tool_name, arguments)


def create_session(server: McpServerConfig):
    if server.type == "stdio":
        return McpStdioSession(server)
    if server.type == "http":
        return McpHttpSession(server)
    raise SkillRouterError(f"Unsupported MCP server type: {server.type}")


class McpStdioSession:
    def __init__(self, server: McpServerConfig):
        self.server = server
        self.process: Optional[subprocess.Popen[bytes]] = None
        self._next_id = 1

    def __enter__(self) -> "McpStdioSession":
        if not self.server.command:
            raise SkillRouterError(f"MCP stdio server {self.server.name!r} has no command")
        env = os.environ.copy()
        env.update(self.server.env)
        self.process = subprocess.Popen(
            [self.server.command, *self.server.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "skill-router", "version": "0.1.0"},
            },
        )
        self._notify("notifications/initialized", {})
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()

    def list_tools(self) -> list[ToolSpec]:
        result = self._request("tools/list", {})
        return tools_from_list_result(self.server.name, result)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = self._request("tools/call", {"name": name, "arguments": arguments})
        if not isinstance(result, dict):
            raise SkillRouterError(f"MCP tool {name!r} returned invalid result")
        return result

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = self._next_id
        self._next_id += 1
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        while True:
            message = self._read()
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise SkillRouterError(f"MCP {method} failed: {message['error']}")
            result = message.get("result", {})
            if not isinstance(result, dict):
                raise SkillRouterError(f"MCP {method} returned non-object result")
            return result

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _send(self, message: dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise SkillRouterError("MCP process is not running")
        body = json.dumps(message, ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        self.process.stdin.write(header + body)
        self.process.stdin.flush()

    def _read(self) -> dict[str, Any]:
        if self.process is None or self.process.stdout is None:
            raise SkillRouterError("MCP process is not running")
        headers: dict[str, str] = {}
        while True:
            line = self.process.stdout.readline()
            if line == b"":
                raise SkillRouterError(f"MCP server {self.server.name!r} closed stdout")
            if line in {b"\r\n", b"\n"}:
                break
            key, _, value = line.decode("ascii").partition(":")
            headers[key.lower()] = value.strip()
        length = int(headers.get("content-length", "0"))
        if length <= 0:
            raise SkillRouterError("MCP message missing Content-Length")
        body = self.process.stdout.read(length)
        return json.loads(body.decode("utf-8"))


class McpHttpSession:
    def __init__(self, server: McpServerConfig):
        self.server = server
        self._next_id = 1
        self._session_id: Optional[str] = None

    def __enter__(self) -> "McpHttpSession":
        self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "skill-router", "version": "0.1.0"},
            },
        )
        self._notify("notifications/initialized", {})
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def list_tools(self) -> list[ToolSpec]:
        result = self._request("tools/list", {})
        return tools_from_list_result(self.server.name, result)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = self._request("tools/call", {"name": name, "arguments": arguments})
        if not isinstance(result, dict):
            raise SkillRouterError(f"MCP tool {name!r} returned invalid result")
        return result

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = self._next_id
        self._next_id += 1
        message = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        response = self._post(message)
        if response.get("id") != request_id:
            raise SkillRouterError(f"MCP HTTP {method} returned mismatched id")
        if "error" in response:
            raise SkillRouterError(f"MCP HTTP {method} failed: {response['error']}")
        result = response.get("result", {})
        if not isinstance(result, dict):
            raise SkillRouterError(f"MCP HTTP {method} returned non-object result")
        return result

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        self._post({"jsonrpc": "2.0", "method": method, "params": params}, allow_empty=True)

    def _post(self, message: dict[str, Any], allow_empty: bool = False) -> dict[str, Any]:
        if not self.server.url:
            raise SkillRouterError(f"MCP HTTP server {self.server.name!r} has no url")
        body = json.dumps(message, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2024-11-05",
        }
        headers.update(self.server.headers)
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        request = urllib.request.Request(
            self.server.url,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                session_id = response.headers.get("Mcp-Session-Id")
                if session_id:
                    self._session_id = session_id
                raw = response.read()
                if allow_empty and not raw:
                    return {}
                content_type = response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise SkillRouterError(f"MCP HTTP error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise SkillRouterError(f"MCP HTTP request failed: {exc.reason}") from exc
        if not raw:
            if allow_empty:
                return {}
            raise SkillRouterError("MCP HTTP response was empty")
        if "text/event-stream" in content_type:
            return parse_sse_json(raw.decode("utf-8"))
        return json.loads(raw.decode("utf-8"))


def schema_to_args(schema: Any) -> dict[str, str]:
    if not isinstance(schema, dict):
        return {}
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return {}
    args: dict[str, str] = {}
    for name, spec in properties.items():
        if not isinstance(name, str) or not isinstance(spec, dict):
            continue
        json_type = spec.get("type", "string")
        if json_type == "integer":
            args[name] = "integer"
        elif json_type == "number":
            args[name] = "number"
        elif json_type == "boolean":
            args[name] = "boolean"
        else:
            args[name] = "string"
    return args


def tools_from_list_result(server_name: str, result: dict[str, Any]) -> list[ToolSpec]:
    raw_tools = result.get("tools", [])
    if not isinstance(raw_tools, list):
        raise SkillRouterError(f"MCP server {server_name!r} returned invalid tools/list")
    tools: list[ToolSpec] = []
    for raw in raw_tools:
        if not isinstance(raw, dict):
            continue
        name = raw.get("name")
        if not isinstance(name, str):
            continue
        tools.append(
            ToolSpec(
                name=name,
                description=str(raw.get("description", "")),
                args=schema_to_args(raw.get("inputSchema", {})),
                source="mcp",
                server=server_name,
            )
        )
    return tools


def normalize_http_url(url: str) -> str:
    cleaned = url.strip()
    if cleaned.startswith("http://http://"):
        cleaned = "http://" + cleaned[len("http://http://") :]
    elif cleaned.startswith("https://https://"):
        cleaned = "https://" + cleaned[len("https://https://") :]
    if not (cleaned.startswith("http://") or cleaned.startswith("https://")):
        raise SkillRouterError(f"MCP HTTP url must start with http:// or https://: {url}")
    return cleaned


def parse_sse_json(text: str) -> dict[str, Any]:
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data and data != "[DONE]":
            parsed = json.loads(data)
            if isinstance(parsed, dict):
                return parsed
    raise SkillRouterError("MCP HTTP SSE response did not contain JSON data")
