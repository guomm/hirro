from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class SkillRouterError(Exception):
    """Base exception for user-facing skill router errors."""


@dataclass(frozen=True)
class CommandSpec:
    id: str
    run: str
    args: dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    directory: Path
    body: str
    commands: dict[str, CommandSpec]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args: dict[str, str] = field(default_factory=dict)
    source: str = "builtin"
    server: Optional[str] = None
