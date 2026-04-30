from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from skill_router.agent import AgentRunner
from skill_router.context import ArtifactStore, ConversationMemory, summarize_payload
from skill_router.mcp import McpConfig, McpToolRegistry
from skill_router.models import CommandSpec, Skill, SkillRouterError, ToolSpec
from skill_router.tools import ToolRegistry, default_tool_registry


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def complete_json(self, messages):
        self.messages.append(messages)
        if not self.responses:
            raise AssertionError("No fake LLM responses left")
        return self.responses.pop(0)


class FakeExecutor:
    def __init__(self):
        self.calls = []

    def execute(self, skill, command_id, arguments):
        self.calls.append((skill, command_id, arguments))
        return subprocess.CompletedProcess(
            args=["fake"],
            returncode=0,
            stdout="skill output\n",
            stderr="",
        )

    def execute_script(self, skill, script_path, args):
        self.calls.append((skill, script_path, args))
        return subprocess.CompletedProcess(
            args=["fake"],
            returncode=0,
            stdout="script output\n",
            stderr="",
        )


class FakeMcpTools:
    def __init__(self):
        self.calls = []
        self.discover_calls = []
        self.loaded_servers = set()
        self._tools = [
            ToolSpec(
                name="precheck_sql",
                description="Precheck SQL",
                args={"sql": "string"},
                source="mcp",
                server="impala-query-mcp",
            )
        ]

    def server_summaries(self):
        return [
            {
                "name": "impala-query-mcp",
                "type": "http",
                "note": "Impala service exposing precheck_sql(sql), exec_sql(sql)",
            }
        ]

    def specs(self):
        return [
            tool
            for tool in self._tools
            if tool.server in self.loaded_servers
        ]

    def specs_for_server(self, server_name):
        self.discover_calls.append(server_name)
        self.loaded_servers.add(server_name)
        return [tool for tool in self._tools if tool.server == server_name]

    def execute(self, server_name, tool_name, arguments):
        self.calls.append((server_name, tool_name, arguments))
        return {"risk": "ok", "decision": "allow"}

    def has_loaded_server(self, server_name):
        return server_name in self.loaded_servers

    def loaded_server_names(self):
        return sorted(self.loaded_servers)


def make_agent(client, skills=None, executor=None, mcp_tools=None) -> AgentRunner:
    return AgentRunner(
        client=client,
        skills=skills or [],
        builtin_tools=default_tool_registry(),
        mcp_tools=mcp_tools or McpToolRegistry(McpConfig([])),
        executor=executor or FakeExecutor(),
    )


def make_skill() -> Skill:
    command = CommandSpec(
        id="echo_text",
        run="py -3.12 scripts/echo.py",
        args={"text": "string"},
        description="Echo text",
    )
    return Skill(
        name="echo",
        description="Echo text",
        directory=Path("."),
        body="Use this skill to echo text.",
        commands={command.id: command},
    )


def make_impala_skill() -> Skill:
    return Skill(
        name="impala-sql-executor",
        description="Use precheck_sql and exec_sql for Impala queries.",
        directory=Path("."),
        body="Use precheck_sql(sql), then exec_sql(sql).",
        commands={},
    )


def make_antispam_skill() -> Skill:
    return Skill(
        name="antispam-ops-skill",
        description="Use anti-spam MCP tools.",
        directory=Path("."),
        body=(
            "Call antispam-ops-mcp.check_antispam_block_status(...), "
            "then antispam-ops-mcp.unblock_antispam_targets(...)."
        ),
        commands={},
    )


class AgentRunnerTests(unittest.TestCase):
    def test_calls_builtin_tool_then_returns_final_answer(self) -> None:
        client = FakeClient(
            [
                {
                    "type": "function_call",
                    "name": "call_tool",
                    "arguments": {
                        "name": "calculator",
                        "args": {"expression": "2 + 3"},
                    },
                },
                {"type": "final", "content": "2 + 3 = 5"},
            ]
        )
        agent = make_agent(client)

        result = agent.run("calculate 2 + 3")

        self.assertEqual(result.content, "2 + 3 = 5")
        self.assertIn('"result": 5', client.messages[1][-1]["content"])
        self.assertEqual(len(result.messages), 3)
        self.assertIn("Turn summary:", result.messages[1]["content"])

    def test_requires_skill_activation_before_execution(self) -> None:
        client = FakeClient(
            [
                {
                    "type": "function_call",
                    "name": "run_skill_command",
                    "arguments": {
                        "skill": "echo",
                        "name": "echo_text",
                        "args": {"text": "hi"},
                    },
                },
                {"type": "final", "content": "I need to activate the skill first."},
            ]
        )
        agent = make_agent(client, skills=[make_skill()])

        result = agent.run("echo hi")

        self.assertEqual(result.content, "I need to activate the skill first.")
        self.assertIn("must be activated", client.messages[1][-1]["content"])

    def test_activates_skill_then_executes_command_then_answers(self) -> None:
        executor = FakeExecutor()
        client = FakeClient(
            [
                {
                    "type": "function_call",
                    "name": "activate_skill",
                    "arguments": {"name": "echo"},
                },
                {
                    "type": "function_call",
                    "name": "run_skill_command",
                    "arguments": {
                        "skill": "echo",
                        "name": "echo_text",
                        "args": {"text": "hi"},
                    },
                },
                {"type": "final", "content": "Printed hi."},
            ]
        )
        agent = make_agent(client, skills=[make_skill()], executor=executor)

        result = agent.run("echo hi")

        self.assertEqual(result.content, "Printed hi.")
        self.assertEqual(executor.calls[0][0].name, "echo")
        self.assertEqual(executor.calls[0][1], "echo_text")
        self.assertEqual(executor.calls[0][2], {"text": "hi"})

    def test_dry_run_stops_at_first_function_call(self) -> None:
        client = FakeClient(
            [
                {
                    "type": "function_call",
                    "name": "call_tool",
                    "arguments": {
                        "name": "calculator",
                        "args": {"expression": "1 + 1"},
                    },
                }
            ]
        )
        agent = make_agent(client)

        result = agent.run("calculate", dry_run=True)

        self.assertIn("Planned function call", result.content)

    def test_activates_skill_then_runs_script_then_answers(self) -> None:
        executor = FakeExecutor()
        client = FakeClient(
            [
                {
                    "type": "function_call",
                    "name": "activate_skill",
                    "arguments": {"name": "echo"},
                },
                {
                    "type": "function_call",
                    "name": "run_skill_script",
                    "arguments": {
                        "skill": "echo",
                        "name": "scripts/helper.py",
                        "args": ["--mode", "retry-plan"],
                    },
                },
                {"type": "final", "content": "Script finished."},
            ]
        )
        agent = make_agent(client, skills=[make_skill()], executor=executor)

        result = agent.run("run helper")

        self.assertEqual(result.content, "Script finished.")
        self.assertEqual(executor.calls[0][1], "scripts/helper.py")
        self.assertEqual(executor.calls[0][2], ["--mode", "retry-plan"])

    def test_function_argument_error_is_returned_to_model_for_repair(self) -> None:
        client = FakeClient(
            [
                {
                    "type": "function_call",
                    "name": "activate_skill",
                    "arguments": {},
                },
                {
                    "type": "function_call",
                    "name": "activate_skill",
                    "arguments": {"name": "echo"},
                },
                {"type": "final", "content": "Activated echo."},
            ]
        )
        agent = make_agent(client, skills=[make_skill()])

        result = agent.run("use echo")

        self.assertEqual(result.content, "Activated echo.")
        all_message_text = "\n".join(
            message["content"]
            for call_messages in client.messages
            for message in call_messages
        )
        self.assertIn("Expected non-empty string field: name", all_message_text)

    def test_invalid_function_call_envelope_is_returned_to_model_for_repair(self) -> None:
        client = FakeClient(
            [
                {
                    "type": "function_call",
                    "arguments": {"name": "echo"},
                },
                {
                    "type": "function_call",
                    "name": "activate_skill",
                    "arguments": {"name": "echo"},
                },
                {"type": "final", "content": "Activated echo."},
            ]
        )
        agent = make_agent(client, skills=[make_skill()])

        result = agent.run("use echo")

        self.assertEqual(result.content, "Activated echo.")
        all_message_text = "\n".join(
            message["content"]
            for call_messages in client.messages
            for message in call_messages
        )
        self.assertIn("__invalid_function_call__", all_message_text)
        self.assertIn("Expected non-empty string field: name", all_message_text)

    def test_call_tool_executes_mcp_tool_with_uniform_fields(self) -> None:
        mcp_tools = FakeMcpTools()
        client = FakeClient(
            [
                {
                    "type": "function_call",
                    "name": "list_tools",
                    "arguments": {"server": "impala-query-mcp"},
                },
                {
                    "type": "function_call",
                    "name": "call_tool",
                    "arguments": {
                        "server": "impala-query-mcp",
                        "name": "precheck_sql",
                        "args": {"sql": "select 1"},
                    },
                },
                {"type": "final", "content": "SQL is ok."},
            ]
        )
        agent = make_agent(client, mcp_tools=mcp_tools)

        result = agent.run("check sql")

        self.assertEqual(result.content, "SQL is ok.")
        self.assertEqual(mcp_tools.calls[0], ("impala-query-mcp", "precheck_sql", {"sql": "select 1"}))

    def test_call_tool_accepts_qualified_server_tool_name(self) -> None:
        mcp_tools = FakeMcpTools()
        client = FakeClient(
            [
                {
                    "type": "function_call",
                    "name": "list_tools",
                    "arguments": {"server": "impala-query-mcp"},
                },
                {
                    "type": "function_call",
                    "name": "call_tool",
                    "arguments": {
                        "name": "impala-query-mcp.precheck_sql",
                        "args": {"sql": "select 1"},
                    },
                },
                {"type": "final", "content": "SQL is ok."},
            ]
        )
        agent = make_agent(client, mcp_tools=mcp_tools)

        result = agent.run("check sql")

        self.assertEqual(result.content, "SQL is ok.")
        self.assertEqual(mcp_tools.calls[0], ("impala-query-mcp", "precheck_sql", {"sql": "select 1"}))

    def test_call_tool_without_server_returns_error_for_unloaded_mcp_tool(self) -> None:
        mcp_tools = FakeMcpTools()
        client = FakeClient(
            [
                {
                    "type": "function_call",
                    "name": "call_tool",
                    "arguments": {
                        "name": "precheck_sql",
                        "args": {"sql": "select 1"},
                    },
                },
                {"type": "final", "content": "Need server or list_tools first."},
            ]
        )
        agent = make_agent(client, mcp_tools=mcp_tools)

        result = agent.run("check sql")

        self.assertEqual(result.content, "Need server or list_tools first.")
        self.assertEqual(mcp_tools.calls, [])
        self.assertIn("server is required", client.messages[1][-1]["content"])

    def test_qualified_mcp_tool_requires_list_tools_first(self) -> None:
        mcp_tools = FakeMcpTools()
        client = FakeClient(
            [
                {
                    "type": "function_call",
                    "name": "call_tool",
                    "arguments": {
                        "name": "impala-query-mcp.precheck_sql",
                        "args": {"sql": "select 1"},
                    },
                },
                {"type": "final", "content": "Need to list tools first."},
            ]
        )
        agent = make_agent(client, mcp_tools=mcp_tools)

        result = agent.run("check sql")

        self.assertEqual(result.content, "Need to list tools first.")
        self.assertEqual(mcp_tools.calls, [])
        self.assertIn("Call list_tools", client.messages[1][-1]["content"])

    def test_list_tools_loads_mcp_tools_on_demand(self) -> None:
        mcp_tools = FakeMcpTools()
        client = FakeClient(
            [
                {
                    "type": "function_call",
                    "name": "list_tools",
                    "arguments": {"server": "impala-query-mcp"},
                },
                {"type": "final", "content": "Loaded tools."},
            ]
        )
        agent = make_agent(client, mcp_tools=mcp_tools)

        result = agent.run("what tools are available?")

        self.assertEqual(result.content, "Loaded tools.")
        self.assertEqual(mcp_tools.discover_calls, ["impala-query-mcp"])
        self.assertIn('"precheck_sql"', client.messages[1][-1]["content"])

    def test_activate_skill_returns_suggested_mcp_servers(self) -> None:
        mcp_tools = FakeMcpTools()
        client = FakeClient(
            [
                {
                    "type": "function_call",
                    "name": "activate_skill",
                    "arguments": {"name": "impala-sql-executor"},
                },
                {"type": "final", "content": "Activated."},
            ]
        )
        agent = make_agent(client, skills=[make_impala_skill()], mcp_tools=mcp_tools)

        result = agent.run("run impala sql")

        self.assertEqual(result.content, "Activated.")
        self.assertIn('"suggested_mcp_servers": ["impala-query-mcp"]', client.messages[1][-1]["content"])
        self.assertIn('"function": "list_tools"', client.messages[1][-1]["content"])

    def test_activate_skill_returns_qualified_tool_refs(self) -> None:
        client = FakeClient(
            [
                {
                    "type": "function_call",
                    "name": "activate_skill",
                    "arguments": {"name": "antispam-ops-skill"},
                },
                {"type": "final", "content": "Activated."},
            ]
        )
        agent = make_agent(client, skills=[make_antispam_skill()], mcp_tools=FakeMcpTools())

        result = agent.run("check spam")

        self.assertEqual(result.content, "Activated.")
        self.assertIn(
            '"server": "antispam-ops-mcp", "name": "check_antispam_block_status"',
            client.messages[1][-1]["content"],
        )

    def test_system_prompt_lists_mcp_servers_not_mcp_tools(self) -> None:
        mcp_tools = FakeMcpTools()
        client = FakeClient([{"type": "final", "content": "ok"}])
        agent = make_agent(client, mcp_tools=mcp_tools)

        agent.run("hello")

        system_prompt = client.messages[0][0]["content"]
        runtime_context = client.messages[0][1]["content"]
        self.assertNotIn('"mcp_servers"', system_prompt)
        self.assertIn("Runtime context:", runtime_context)
        self.assertIn('"mcp_servers"', runtime_context)
        self.assertIn("impala-query-mcp", runtime_context)
        self.assertNotIn('"mcp_tools"', runtime_context)
        self.assertIn('"loaded_tools": []', runtime_context)
        self.assertIn('"loaded_servers": []', runtime_context)
        self.assertIn('"activated_skills": []', runtime_context)
        self.assertEqual(mcp_tools.discover_calls, [])

    def test_runtime_context_includes_loaded_tool_minimal_schema_after_list_tools(self) -> None:
        mcp_tools = FakeMcpTools()
        first_client = FakeClient(
            [
                {
                    "type": "function_call",
                    "name": "list_tools",
                    "arguments": {"server": "impala-query-mcp"},
                },
                {"type": "final", "content": "Loaded tools."},
            ]
        )
        agent = make_agent(first_client, mcp_tools=mcp_tools)
        agent.run("load tools")

        second_client = FakeClient([{"type": "final", "content": "ok"}])
        agent.client = second_client
        agent.run("next turn")

        runtime_context = second_client.messages[0][1]["content"]
        self.assertIn('"loaded_servers": ["impala-query-mcp"]', runtime_context)
        self.assertIn('"loaded_tools": [{"server": "impala-query-mcp", "name": "precheck_sql", "args": {"sql": "string"}}]', runtime_context)

    def test_runtime_context_includes_activated_skills(self) -> None:
        client = FakeClient(
            [
                {
                    "type": "function_call",
                    "name": "activate_skill",
                    "arguments": {"name": "echo"},
                },
                {"type": "final", "content": "Activated."},
            ]
        )
        agent = make_agent(client, skills=[make_skill()])
        agent.run("use echo")

        next_client = FakeClient([{"type": "final", "content": "ok"}])
        agent.client = next_client
        agent.run("next turn")

        runtime_context = next_client.messages[0][1]["content"]
        self.assertIn('"activated_skills": ["echo"]', runtime_context)

    def test_large_tool_result_is_compacted_into_artifact(self) -> None:
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="big_result",
                description="Return a large payload.",
                args={"text": "string"},
            ),
            lambda arguments: {"blob": arguments["text"] * 800},
        )
        client = FakeClient(
            [
                {
                    "type": "function_call",
                    "name": "call_tool",
                    "arguments": {
                        "name": "big_result",
                        "args": {"text": "abcdef"},
                    },
                },
                {"type": "final", "content": "Stored the large result."},
            ]
        )
        with TemporaryDirectory() as raw:
            agent = AgentRunner(
                client=client,
                skills=[],
                builtin_tools=registry,
                mcp_tools=McpToolRegistry(McpConfig([])),
                executor=FakeExecutor(),
                artifact_store=ArtifactStore(Path(raw)),
                max_inline_result_chars=200,
            )

            result = agent.run("generate large output")

        self.assertEqual(result.content, "Stored the large result.")
        self.assertIn("artifact=", result.messages[1]["content"])
        self.assertIn('"artifact_id": "call_tool-', client.messages[1][-1]["content"])
        self.assertIn('"full_result_omitted": true', client.messages[1][-1]["content"])

    def test_generic_mcp_payload_summary_keeps_high_signal_fields(self) -> None:
        summary = summarize_payload(
            {
                "ok": True,
                "message": "query executed successfully",
                "data": [
                    {"dt": "2026-04-29", "count": 12, "table": "dwd.demo"},
                    {"dt": "2026-04-28", "count": 9, "table": "dwd.demo"},
                ],
                "meta": {"row_count": 2, "elapsed_ms": 31},
            }
        )

        self.assertIn("ok=True", summary)
        self.assertIn("message=query executed successfully", summary)
        self.assertIn("data=list[2]{dt, count, table}", summary)
        self.assertIn("meta{row_count=2", summary)

    def test_conversation_memory_rolls_old_turns_into_summary(self) -> None:
        memory = ConversationMemory(max_turns=1, keep_last_turns=1)
        memory.append_turn(
            [
                {"role": "user", "content": "first task"},
                {"role": "assistant", "content": "Turn summary:\n- did first"},
                {"role": "assistant", "content": "first answer"},
            ]
        )
        memory.append_turn(
            [
                {"role": "user", "content": "second task"},
                {"role": "assistant", "content": "Turn summary:\n- did second"},
                {"role": "assistant", "content": "second answer"},
            ]
        )

        history = memory.export()

        self.assertEqual(history[0]["role"], "assistant")
        self.assertIn("Conversation summary:", history[0]["content"])
        self.assertIn("user=first task", history[0]["content"])
        self.assertIn("steps=- did first", history[0]["content"])
        self.assertIn("answer=first answer", history[0]["content"])
        self.assertIn("second answer", history[-1]["content"])

    def test_conversation_memory_prefers_llm_summary_after_threshold(self) -> None:
        captured_turns = []
        logs = []

        def summarizer(turns):
            captured_turns.append(turns)
            return "llm summary: task one completed, task two pending"

        memory = ConversationMemory(
            max_turns=2,
            keep_last_turns=1,
            summarizer=summarizer,
            logger=logs.append,
        )
        memory.append_turn(
            [
                {"role": "user", "content": "task one"},
                {"role": "assistant", "content": "Turn summary:\n- did one"},
                {"role": "assistant", "content": "done one"},
            ]
        )
        memory.append_turn(
            [
                {"role": "user", "content": "task two"},
                {"role": "assistant", "content": "Turn summary:\n- did two"},
                {"role": "assistant", "content": "done two"},
            ]
        )
        memory.append_turn(
            [
                {"role": "user", "content": "task three"},
                {"role": "assistant", "content": "Turn summary:\n- did three"},
                {"role": "assistant", "content": "done three"},
            ]
        )

        history = memory.export()

        self.assertEqual(len(captured_turns), 1)
        self.assertIn("llm summary: task one completed, task two pending", history[0]["content"])
        self.assertEqual(len(logs), 1)
        self.assertIn("context compression triggered", logs[0])
        self.assertIn("using llm summary", logs[0])


if __name__ == "__main__":
    unittest.main()
