from __future__ import annotations

from pathlib import Path
from typing import Any

from skill_router.frontmatter import split_frontmatter
from skill_router.models import CommandSpec, Skill, SkillRouterError


class SkillLoader:
    def __init__(self, skills_dir: Path | str):
        self.skills_dir = Path(skills_dir)

    def load(self) -> list[Skill]:
        if not self.skills_dir.exists():
            raise SkillRouterError(f"Skills directory does not exist: {self.skills_dir}")
        if not self.skills_dir.is_dir():
            raise SkillRouterError(f"Skills path is not a directory: {self.skills_dir}")

        skills: list[Skill] = []
        for child in sorted(self.skills_dir.iterdir()):
            if not child.is_dir():
                continue
            skill_md = child / "SKILL.md"
            if not skill_md.exists():
                continue
            skills.append(self._load_one(child, skill_md))

        if not skills:
            raise SkillRouterError(f"No skills with SKILL.md found in: {self.skills_dir}")
        return skills

    def load_optional(self) -> list[Skill]:
        if not self.skills_dir.exists():
            return []
        if not self.skills_dir.is_dir():
            raise SkillRouterError(f"Skills path is not a directory: {self.skills_dir}")
        skills: list[Skill] = []
        for child in sorted(self.skills_dir.iterdir()):
            if not child.is_dir():
                continue
            skill_md = child / "SKILL.md"
            if skill_md.exists():
                skills.append(self._load_one(child, skill_md))
        return skills

    def _load_one(self, directory: Path, skill_md: Path) -> Skill:
        metadata, body = split_frontmatter(skill_md.read_text(encoding="utf-8"))
        name = _require_string(metadata, "name", skill_md)
        description = str(metadata.get("description", "")).strip()
        commands = _parse_commands(metadata.get("command"), skill_md)
        return Skill(
            name=name,
            description=description,
            directory=directory,
            body=body.strip(),
            commands=commands,
        )


def _parse_commands(value: Any, source: Path) -> dict[str, CommandSpec]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SkillRouterError(f"{source} must declare a command map in frontmatter")

    command_id = _require_string(value, "id", source)
    run = _require_string(value, "run", source)
    args_value = value.get("args", {})
    if args_value is None:
        args_value = {}
    if not isinstance(args_value, dict):
        raise SkillRouterError(f"{source} command.args must be a map")

    args: dict[str, str] = {}
    for key, arg_type in args_value.items():
        if not isinstance(key, str) or not isinstance(arg_type, str):
            raise SkillRouterError(f"{source} command args must map string names to string types")
        if arg_type not in {"string", "integer", "number", "boolean"}:
            raise SkillRouterError(
                f"{source} argument {key!r} has unsupported type {arg_type!r}"
            )
        args[key] = arg_type

    spec = CommandSpec(
        id=command_id,
        run=run,
        args=args,
        description=str(value.get("description", "")).strip(),
    )
    return {spec.id: spec}


def _require_string(data: dict[str, Any], key: str, source: Path) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SkillRouterError(f"{source} must declare a non-empty string field: {key}")
    return value.strip()
