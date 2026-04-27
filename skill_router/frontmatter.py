from __future__ import annotations

from typing import Any, Tuple, Union

from skill_router.models import SkillRouterError


def split_frontmatter(text: str) -> Tuple[dict[str, Any], str]:
    normalized = text.replace("\r\n", "\n")
    if not normalized.startswith("---\n"):
        return {}, text

    end = normalized.find("\n---\n", 4)
    if end == -1:
        raise SkillRouterError("SKILL.md frontmatter starts with --- but has no closing ---")

    raw = normalized[4:end]
    body = normalized[end + len("\n---\n") :]
    return parse_simple_yaml(raw), body


def parse_simple_yaml(raw: str) -> dict[str, Any]:
    """Parse the small YAML subset used by SKILL.md metadata.

    Supported forms are intentionally narrow: string scalars and nested maps
    by indentation. This avoids a runtime PyYAML dependency while keeping the
    command manifest structured.
    """

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for line_number, original in enumerate(raw.splitlines(), start=1):
        if not original.strip() or original.lstrip().startswith("#"):
            continue
        if "\t" in original[: len(original) - len(original.lstrip(" \t"))]:
            raise SkillRouterError(f"Tabs are not supported in frontmatter line {line_number}")

        indent = len(original) - len(original.lstrip(" "))
        stripped = original.strip()
        if ":" not in stripped:
            raise SkillRouterError(f"Invalid frontmatter line {line_number}: {stripped}")

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise SkillRouterError(f"Empty frontmatter key on line {line_number}")

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)

    return root


def _parse_scalar(value: str) -> Union[str, bool, int, float]:
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
