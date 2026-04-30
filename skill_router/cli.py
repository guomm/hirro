from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skill_router.agent import AgentRunner
from skill_router.context import ArtifactStore, ConversationMemory
from skill_router.executor import Executor
from skill_router.llm import LLMClient
from skill_router.loader import SkillLoader
from skill_router.mcp import McpConfig, McpToolRegistry
from skill_router.models import SkillRouterError
from skill_router.tools import default_tool_registry
from skill_router.whitelist import WhitelistStore


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Route and execute local skills safely."
    )
    parser.add_argument(
        "task",
        nargs="?",
        help="Natural-language task to route. Omit it or pass --chat for interactive mode.",
    )
    parser.add_argument(
        "--skills-dir", default="./skills", help="Directory containing skill folders."
    )
    parser.add_argument(
        "--model",
        default="deepseek-v4-flash",
        # required=True,
        help="OpenAI-compatible model name.",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.deepseek.com",
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--whitelist",
        default=".skill-router/whitelist.json",
        help="Project-local whitelist path.",
    )
    parser.add_argument(
        "--mcp-config",
        default=".mcp.json",
        help="MCP server config path. Missing config disables MCP tools.",
    )
    parser.add_argument(
        "--plan-only", action="store_true", help="Print the plan without executing."
    )
    parser.add_argument(
        "--chat", action="store_true", help="Start an interactive multi-turn chat."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every LLM request message list and JSON response.",
    )
    args = parser.parse_args(argv)

    try:
        skills = SkillLoader(Path(args.skills_dir)).load_optional()
        artifact_store = ArtifactStore(Path(".skill-router") / "artifacts")
        builtin_tools = default_tool_registry(artifact_store=artifact_store)
        mcp_tools = McpToolRegistry(McpConfig.load_optional(args.mcp_config))
        client = LLMClient(
            model=args.model,
            base_url=args.base_url,
            api_key="sk-6c1cf3350fc94e3e8f3e6a9fa7ef6684",
        )
        executor = Executor(WhitelistStore(args.whitelist))
        agent = AgentRunner(
            client=client,
            skills=skills,
            builtin_tools=builtin_tools,
            mcp_tools=mcp_tools,
            executor=executor,
            verbose=args.verbose,
            artifact_store=artifact_store,
        )

        if args.chat or args.task is None:
            return _run_chat(agent=agent, plan_only=args.plan_only)

        result = agent.run(args.task, history=[], dry_run=args.plan_only)
        print(result.content)
        return 0
    except SkillRouterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run_chat(
    agent: AgentRunner,
    plan_only: bool,
) -> int:
    memory = ConversationMemory(
        summarizer=_make_memory_summarizer(agent.client),
        logger=print if agent.verbose else None,
    )
    print("Skill Router chat. Type /exit or /quit to stop.")
    while True:
        try:
            task = input("> ").strip()
        except EOFError:
            print()
            return 0
        if not task:
            continue
        if task in {"/exit", "/quit"}:
            return 0
        try:
            result = agent.run(task, history=memory.export(), dry_run=plan_only)
            print(result.content)
            memory.append_turn(result.messages)
        except SkillRouterError as exc:
            print(f"error: {exc}", file=sys.stderr)
            memory.append_error(task, str(exc))


def _make_memory_summarizer(client: LLMClient):
    def summarize(turns: list[list[dict[str, str]]]) -> str:
        transcript_lines: list[str] = []
        for index, turn in enumerate(turns, start=1):
            transcript_lines.append(f"Turn {index}:")
            for message in turn:
                role = message.get("role", "assistant")
                content = message.get("content", "")
                transcript_lines.append(f"{role}: {content}")
        messages = [
            {
                "role": "system",
                "content": (
                    "Summarize old chat turns for future continuation. "
                    "Return exactly one JSON object with a string field 'summary'. "
                    "Focus on user goals, completed actions, important results, active skills, "
                    "loaded servers or tools when relevant, and pending follow-ups. "
                    "Keep it compact and factual."
                ),
            },
            {
                "role": "user",
                "content": "\n".join(transcript_lines),
            },
        ]
        response = client.complete_json(messages)
        summary = response.get("summary", "")
        if not isinstance(summary, str):
            return ""
        return " ".join(summary.split()).strip()

    return summarize


if __name__ == "__main__":
    raise SystemExit(main())
