from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from skill_router.models import SkillRouterError


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    path: Path


class ArtifactStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._counter = 0

    def save(self, payload: Any, prefix: str = "artifact") -> ArtifactRecord:
        self._counter += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        artifact_id = f"{prefix}-{timestamp}-{self._counter:04d}"
        path = self.root / f"{artifact_id}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return ArtifactRecord(artifact_id=artifact_id, path=path)

    def read_text(self, artifact_id: str) -> str:
        path = self.root / f"{artifact_id}.json"
        if not path.exists():
            raise SkillRouterError(f"Unknown artifact: {artifact_id}")
        return path.read_text(encoding="utf-8")


class ConversationMemory:
    def __init__(
        self,
        max_turns: int = 10,
        keep_last_turns: int = 3,
        summarizer: Optional[Callable[[list[list[dict[str, str]]]], str]] = None,
        logger: Optional[Callable[[str], None]] = None,
    ):
        self.max_turns = max_turns
        self.keep_last_turns = keep_last_turns
        self.summarizer = summarizer
        self.logger = logger
        self._recent_turns: list[list[dict[str, str]]] = []
        self._summaries: list[str] = []

    def export(self) -> list[dict[str, str]]:
        recent = [message for turn in self._recent_turns for message in turn]
        if not self._summaries:
            return recent
        summary = "Conversation summary:\n" + "\n".join(
            f"- {line}" for line in self._summaries
        )
        return [{"role": "assistant", "content": summary}, *recent]

    def append_turn(self, messages: list[dict[str, str]]) -> None:
        self._recent_turns.append(list(messages))
        self._compact()

    def append_error(self, task: str, error: str) -> None:
        self.append_turn(
            [
                {"role": "user", "content": task},
                {"role": "assistant", "content": f"Error: {error}"},
            ]
        )

    def _compact(self) -> None:
        if len(self._recent_turns) <= self.max_turns:
            return
        overflow = len(self._recent_turns) - self.keep_last_turns
        if overflow <= 0:
            return
        head = self._recent_turns[:overflow]
        self._recent_turns = self._recent_turns[overflow:]
        summary, source = self._summarize_compacted_turns(head)
        if summary:
            self._summaries.append(summary)
            if self.logger is not None:
                self.logger(
                    f"context compression triggered: compacted {len(head)} old turns using {source} summary; "
                    f"kept {len(self._recent_turns)} recent turns"
                )

    def _summarize_compacted_turns(self, turns: list[list[dict[str, str]]]) -> tuple[str, str]:
        if self.summarizer is not None:
            try:
                summary = self.summarizer(turns)
            except Exception:
                summary = ""
            if summary:
                return summary, "llm"
        return (
            " | ".join(summary for summary in (summarize_turn(turn) for turn in turns) if summary),
            "rule",
        )


class TurnCompressor:
    def __init__(
        self,
        user_input: str,
        artifact_store: Optional[ArtifactStore] = None,
        max_inline_result_chars: int = 4000,
    ):
        self.user_input = user_input
        self.artifact_store = artifact_store
        self.max_inline_result_chars = max_inline_result_chars
        self._step_summaries: list[str] = []

    def note_invalid_function_call(self, error: Exception) -> None:
        self._step_summaries.append(f"invalid function call repaired: {error}")

    def record_dry_run(self, call_text: str) -> list[dict[str, str]]:
        return build_turn_messages(self.user_input, [call_text], f"Planned function call: {call_text}")

    def record_function_result(
        self,
        function_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        compacted = compact_function_result(
            function_name,
            result,
            artifact_store=self.artifact_store,
            max_inline_chars=self.max_inline_result_chars,
        )
        self._step_summaries.append(describe_step(function_name, arguments, compacted))
        return compacted

    def build_turn_messages(self, final_content: str) -> list[dict[str, str]]:
        return build_turn_messages(self.user_input, self._step_summaries, final_content)

def summarize_turn(messages: list[dict[str, str]]) -> str:
    user_input = ""
    turn_summary = ""
    final_answer = ""
    for message in messages:
        content = message.get("content", "")
        if message.get("role") == "user" and not user_input:
            user_input = _shorten(content, 120)
        elif message.get("role") == "assistant" and content.startswith("Turn summary:\n"):
            turn_summary = _shorten(
                content.removeprefix("Turn summary:\n").replace("\n", " | "),
                240,
            )
        elif message.get("role") == "assistant":
            final_answer = _shorten(content, 160)
    parts = []
    if user_input:
        parts.append(f"user={user_input}")
    if turn_summary:
        parts.append(f"steps={turn_summary}")
    if final_answer:
        parts.append(f"answer={final_answer}")
    return "; ".join(parts)


def compact_function_result(
    function_name: str,
    result: dict[str, Any],
    artifact_store: Optional[ArtifactStore],
    max_inline_chars: int,
) -> dict[str, Any]:
    raw_text = json.dumps(result, ensure_ascii=False)
    if function_name in {"activate_skill", "list_tools", "list_capabilities"}:
        return result
    if len(raw_text) <= max_inline_chars:
        return result

    record = artifact_store.save(result, prefix=function_name) if artifact_store else None
    compacted = {
        "ok": _infer_ok(result),
        "summary": summarize_payload(result),
        "preview": build_preview(result),
        "full_result_omitted": True,
    }
    if record is not None:
        compacted["artifact_id"] = record.artifact_id
    return compacted


def summarize_payload(payload: Any) -> str:
    if isinstance(payload, dict):
        parts = summarize_dict(payload)
        if parts:
            return "; ".join(parts[:6])
        keys = ", ".join(sorted(str(key) for key in payload.keys())[:8])
        return f"dict keys: {keys}"
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict):
            columns = ", ".join(str(key) for key in list(payload[0].keys())[:6])
            return f"list[{len(payload)}] of objects: {columns}"
        return f"list with {len(payload)} items"
    if isinstance(payload, str):
        return _shorten(payload, 160)
    return _shorten(str(payload), 160)


def build_preview(payload: Any) -> Any:
    if isinstance(payload, dict):
        return _prune_value(payload, depth=2, max_items=4, string_limit=200)
    if isinstance(payload, list):
        return _prune_value(payload, depth=2, max_items=3, string_limit=200)
    if isinstance(payload, str):
        return _shorten(payload, 200)
    return payload


def describe_step(function_name: str, arguments: dict[str, Any], result: dict[str, Any]) -> str:
    if function_name == "activate_skill":
        return f"activated skill {arguments.get('name', '?')}"
    if function_name == "list_tools":
        server = arguments.get("server", "all")
        tools = result.get("tools")
        if isinstance(tools, list):
            names = [tool.get("name", "?") for tool in tools[:5] if isinstance(tool, dict)]
            return f"loaded tools for {server}: {', '.join(names)}"
        return f"listed tools for {server}"
    if function_name == "call_tool":
        tool_name = arguments.get("name", "?")
        summary = result.get("summary") or summarize_payload(result)
        artifact_id = result.get("artifact_id")
        if artifact_id:
            return f"called tool {tool_name}, summary: {summary}, artifact={artifact_id}"
        return f"called tool {tool_name}, summary: {summary}"
    if function_name == "run_skill_command":
        return f"ran skill command {arguments.get('skill', '?')}.{arguments.get('name', '?')}"
    if function_name == "run_skill_script":
        return f"ran skill script {arguments.get('skill', '?')}.{arguments.get('name', '?')}"
    return f"{function_name}: {summarize_payload(result)}"


def build_turn_messages(
    user_input: str,
    step_summaries: list[str],
    final_content: str,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "user", "content": user_input}]
    if step_summaries:
        messages.append(
            {
                "role": "assistant",
                "content": "Turn summary:\n" + "\n".join(f"- {line}" for line in step_summaries),
            }
        )
    messages.append({"role": "assistant", "content": final_content})
    return messages


def _infer_ok(result: dict[str, Any]) -> bool:
    if "ok" in result and isinstance(result["ok"], bool):
        return result["ok"]
    return True


def _shorten(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def summarize_dict(payload: dict[str, Any]) -> list[str]:
    preferred_keys = [
        "ok",
        "status",
        "decision",
        "message",
        "error",
        "risk",
        "returncode",
        "count",
        "total",
        "row_count",
        "affected_rows",
    ]
    parts: list[str] = []
    for key in preferred_keys:
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, (str, int, float, bool)) and value != "":
            parts.append(f"{key}={_shorten(str(value), 80)}")

    for key, value in payload.items():
        if key in preferred_keys:
            continue
        if isinstance(value, str) and value.strip():
            parts.append(f"{key}={_shorten(value, 100)}")
        elif isinstance(value, list):
            parts.append(describe_list(key, value))
        elif isinstance(value, dict):
            nested = summarize_dict(value)
            if nested:
                parts.append(f"{key}{{{'; '.join(nested[:3])}}}")
            else:
                parts.append(f"{key}=object[{len(value)}]")
    return parts


def describe_list(key: str, value: list[Any]) -> str:
    if not value:
        return f"{key}=list[0]"
    first = value[0]
    if isinstance(first, dict):
        columns = ", ".join(str(name) for name in list(first.keys())[:6])
        return f"{key}=list[{len(value)}]{{{columns}}}"
    if isinstance(first, str):
        return f"{key}=list[{len(value)}]({_shorten(first, 60)})"
    return f"{key}=list[{len(value)}]"


def _prune_value(value: Any, depth: int, max_items: int, string_limit: int) -> Any:
    if isinstance(value, str):
        return _shorten(value, string_limit)
    if isinstance(value, list):
        if depth <= 0:
            return f"list[{len(value)}]"
        return [
            _prune_value(item, depth - 1, max_items, string_limit)
            for item in value[:max_items]
        ]
    if isinstance(value, dict):
        if depth <= 0:
            return f"object[{len(value)}]"
        return {
            key: _prune_value(item, depth - 1, max_items, string_limit)
            for key, item in list(value.items())[:max_items]
        }
    return value
