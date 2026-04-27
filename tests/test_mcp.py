from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from skill_router.mcp import (
    McpConfig,
    McpHttpSession,
    McpServerConfig,
    McpToolRegistry,
    parse_sse_json,
    schema_to_args,
)


class McpTests(unittest.TestCase):
    def test_loads_mcp_config(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "mcp.json"
            path.write_text(
                json.dumps(
                    {
                        "servers": {
                            "demo": {
                                "command": "py",
                                "args": ["-3.12", "server.py"],
                                "env": {"TOKEN": "x"},
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            config = McpConfig.load_optional(path)

            self.assertIn("demo", config.servers)
            self.assertEqual(config.servers["demo"].command, "py")
            self.assertEqual(config.servers["demo"].args, ["-3.12", "server.py"])

    def test_loads_http_mcp_servers_config(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / ".mcp.json"
            path.write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "impala-mcp": {
                                "type": "http",
                                "headers": {
                                    "Authorization": "Bearer token",
                                    "X-Tenant": "dev",
                                },
                                "url": "http://http://bi-metabase.vvic.com:8686/mcp/impala-query/",
                                "note": "Impala tools",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            config = McpConfig.load_optional(path)

            self.assertIn("impala-mcp", config.servers)
            self.assertEqual(config.servers["impala-mcp"].type, "http")
            self.assertEqual(
                config.servers["impala-mcp"].url,
                "http://bi-metabase.vvic.com:8686/mcp/impala-query/",
            )
            self.assertEqual(
                config.servers["impala-mcp"].headers,
                {"Authorization": "Bearer token", "X-Tenant": "dev"},
            )
            self.assertEqual(config.servers["impala-mcp"].note, "Impala tools")

    def test_http_session_lists_tools(self) -> None:
        server = McpConfig(
            [
                # Reuse config loading validation for construction shape.
            ]
        )
        del server
        with TemporaryDirectory() as raw:
            path = Path(raw) / ".mcp.json"
            path.write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "demo": {
                                "type": "http",
                                "url": "http://example.test/mcp",
                                "headers": {"Authorization": "Bearer token"},
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            config = McpConfig.load_optional(path)
            responses = [
                _FakeHttpResponse(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "result": {"protocolVersion": "2024-11-05"},
                    },
                    headers={"Mcp-Session-Id": "session-1"},
                ),
                _FakeHttpResponse({}, raw=b""),
                _FakeHttpResponse(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "result": {
                            "tools": [
                                {
                                    "name": "exec_sql",
                                    "description": "Execute SQL",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {"sql": {"type": "string"}},
                                    },
                                }
                            ]
                        },
                    }
                ),
            ]
            requests = []

            def fake_urlopen(request, timeout):
                requests.append(request)
                return responses.pop(0)

            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                with McpHttpSession(config.servers["demo"]) as session:
                    tools = session.list_tools()

            self.assertEqual(tools[0].name, "exec_sql")
            self.assertEqual(tools[0].server, "demo")
            self.assertEqual(tools[0].args, {"sql": "string"})
            self.assertEqual(requests[0].full_url, "http://example.test/mcp")
            self.assertEqual(requests[0].headers["Authorization"], "Bearer token")
            self.assertEqual(requests[2].headers["Mcp-session-id"], "session-1")

    def test_parse_sse_json(self) -> None:
        parsed = parse_sse_json(
            'event: message\n'
            'data: {"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n\n'
        )

        self.assertEqual(parsed["result"], {"ok": True})

    def test_registry_remembers_loaded_server_even_when_no_tools(self) -> None:
        registry = McpToolRegistry(
            McpConfig([McpServerConfig(name="empty", command="fake")])
        )
        session = _FakeMcpSession([])

        with mock.patch("skill_router.mcp.create_session", return_value=session):
            first = registry.specs_for_server("empty")
            second = registry.specs_for_server("empty")

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertTrue(registry.has_loaded_server("empty"))
        self.assertEqual(session.list_tools_calls, 1)

    def test_schema_to_args(self) -> None:
        args = schema_to_args(
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                    "exact": {"type": "boolean"},
                },
            }
        )

        self.assertEqual(args, {"query": "string", "limit": "integer", "exact": "boolean"})


if __name__ == "__main__":
    unittest.main()


class _FakeHttpResponse:
    def __init__(self, payload, headers=None, raw=None):
        self.headers = headers or {"Content-Type": "application/json"}
        self._raw = raw if raw is not None else json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def read(self):
        return self._raw


class _FakeMcpSession:
    def __init__(self, tools):
        self.tools = tools
        self.list_tools_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def list_tools(self):
        self.list_tools_calls += 1
        return self.tools
