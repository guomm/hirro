from __future__ import annotations

import ast
import operator
from datetime import datetime
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

from skill_router.context import ArtifactStore
from skill_router.executor import validate_schema_arguments
from skill_router.models import SkillRouterError, ToolSpec


ToolFunction = Callable[[dict[str, Any]], dict[str, Any]]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolSpec, ToolFunction]] = {}

    def register(self, spec: ToolSpec, func: ToolFunction) -> None:
        if spec.name in self._tools:
            raise SkillRouterError(f"Duplicate tool registered: {spec.name}")
        self._tools[spec.name] = (spec, func)

    def specs(self) -> list[ToolSpec]:
        return [entry[0] for entry in self._tools.values()]

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        entry = self._tools.get(tool_name)
        if entry is None:
            raise SkillRouterError(f"Unknown built-in tool: {tool_name}")
        spec, func = entry
        arguments = validate_schema_arguments(spec.args, arguments)
        return func(arguments)


def default_tool_registry(artifact_store: Optional[ArtifactStore] = None) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="calculator",
            description="Evaluate a basic arithmetic expression with +, -, *, /, %, ** and parentheses.",
            args={"expression": "string"},
        ),
        _calculator,
    )
    registry.register(
        ToolSpec(
            name="current_time",
            description="Return the current time for an IANA timezone.",
            args={"timezone": "string"},
        ),
        _current_time,
    )
    registry.register(
        ToolSpec(
            name="text_stats",
            description="Count characters, words, and lines in text.",
            args={"text": "string"},
        ),
        _text_stats,
    )
    if artifact_store is not None:
        registry.register(
            ToolSpec(
                name="read_artifact",
                description="Read a stored large result artifact by id.",
                args={"artifact_id": "string", "max_chars": "integer"},
            ),
            lambda arguments: _read_artifact(artifact_store, arguments),
        )
    return registry


def _calculator(arguments: dict[str, Any]) -> dict[str, Any]:
    expression = arguments["expression"]
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_arithmetic(tree.body)
    except (SyntaxError, ValueError, ZeroDivisionError) as exc:
        raise SkillRouterError(f"Invalid calculator expression: {exc}") from exc
    return {"result": result}


def _current_time(arguments: dict[str, Any]) -> dict[str, Any]:
    timezone = arguments["timezone"]
    try:
        now = datetime.now(ZoneInfo(timezone))
    except Exception as exc:
        raise SkillRouterError(f"Invalid timezone: {timezone}") from exc
    return {"timezone": timezone, "iso": now.isoformat(timespec="seconds")}


def _text_stats(arguments: dict[str, Any]) -> dict[str, Any]:
    text = arguments["text"]
    return {
        "characters": len(text),
        "words": len(text.split()),
        "lines": 0 if text == "" else text.count("\n") + 1,
    }


def _read_artifact(store: ArtifactStore, arguments: dict[str, Any]) -> dict[str, Any]:
    artifact_id = arguments["artifact_id"]
    max_chars = arguments["max_chars"]
    if max_chars <= 0:
        raise SkillRouterError("read_artifact.max_chars must be greater than 0")
    content = store.read_text(artifact_id)
    truncated = len(content) > max_chars
    return {
        "artifact_id": artifact_id,
        "content": content[:max_chars],
        "truncated": truncated,
    }


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_arithmetic(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _eval_arithmetic(node.left)
        right = _eval_arithmetic(node.right)
        return _BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_eval_arithmetic(node.operand))
    raise ValueError("only arithmetic expressions are allowed")
