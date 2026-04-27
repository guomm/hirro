from __future__ import annotations

import shlex
import subprocess
from typing import Any, Callable, Optional

from skill_router.models import CommandSpec, Skill, SkillRouterError
from skill_router.whitelist import WhitelistEntry, WhitelistStore


Confirmer = Callable[[Skill, CommandSpec, Any], bool]


class Executor:
    def __init__(
        self,
        whitelist: WhitelistStore,
        confirmer: Optional[Confirmer] = None,
    ):
        self.whitelist = whitelist
        self.confirmer = confirmer or _default_confirmer

    def execute(
        self,
        skill: Skill,
        command_id: str,
        arguments: dict[str, Any],
    ) -> subprocess.CompletedProcess[str]:
        command = skill.commands.get(command_id)
        if command is None:
            raise SkillRouterError(
                f"Unknown command {command_id!r} for skill {skill.name!r}"
            )
        arguments = validate_arguments(command, arguments)
        entry = WhitelistEntry.from_command(skill, command)
        if not self.whitelist.contains(entry):
            if not self.confirmer(skill, command, arguments):
                raise SkillRouterError("Execution cancelled by user")
            self.whitelist.add(entry)

        argv = build_argv(command, arguments)
        return subprocess.run(
            argv,
            cwd=skill.directory,
            shell=False,
            check=False,
            text=True,
            capture_output=True,
        )

    def execute_script(
        self,
        skill: Skill,
        script_path: str,
        args: list[str],
        python: str = "py -3.12",
    ) -> subprocess.CompletedProcess[str]:
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            raise SkillRouterError("Script args must be a string list")
        script = _validate_skill_script_path(skill, script_path)
        command = CommandSpec(
            id=f"script:{script.as_posix()}",
            run=f"{python} {script.as_posix()}",
            args={},
            description=f"Run skill script {script.as_posix()}",
        )
        entry = WhitelistEntry.from_command(skill, command)
        if not self.whitelist.contains(entry):
            if not self.confirmer(skill, command, args):
                raise SkillRouterError("Execution cancelled by user")
            self.whitelist.add(entry)

        argv = [*shlex.split(python, posix=False), script.as_posix(), *args]
        return subprocess.run(
            argv,
            cwd=skill.directory,
            shell=False,
            check=False,
            text=True,
            capture_output=True,
        )


def validate_arguments(command: CommandSpec, arguments: dict[str, Any]) -> dict[str, Any]:
    return validate_schema_arguments(command.args, arguments)


def validate_schema_arguments(schema: dict[str, str], arguments: dict[str, Any]) -> dict[str, Any]:
    expected = set(schema)
    provided = set(arguments)
    missing = expected - provided
    extra = provided - expected
    if missing:
        raise SkillRouterError(f"Missing command arguments: {', '.join(sorted(missing))}")
    if extra:
        raise SkillRouterError(f"Unexpected command arguments: {', '.join(sorted(extra))}")

    validated: dict[str, Any] = {}
    for name, expected_type in schema.items():
        value = arguments[name]
        if expected_type == "string":
            if not isinstance(value, str):
                raise SkillRouterError(f"Argument {name!r} must be a string")
            validated[name] = value
        elif expected_type == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise SkillRouterError(f"Argument {name!r} must be an integer")
            validated[name] = value
        elif expected_type == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SkillRouterError(f"Argument {name!r} must be a number")
            validated[name] = value
        elif expected_type == "boolean":
            if not isinstance(value, bool):
                raise SkillRouterError(f"Argument {name!r} must be a boolean")
            validated[name] = value
        else:
            raise SkillRouterError(f"Unsupported argument type: {expected_type}")
    return validated


def build_argv(command: CommandSpec, arguments: dict[str, Any]) -> list[str]:
    argv = shlex.split(command.run, posix=False)
    if not argv:
        raise SkillRouterError("Command run template is empty")
    for name in command.args:
        value = arguments[name]
        argv.extend([f"--{name.replace('_', '-')}", str(value)])
    return argv


def _validate_skill_script_path(skill: Skill, script_path: str):
    if not isinstance(script_path, str) or not script_path.strip():
        raise SkillRouterError("script_path must be a non-empty string")
    candidate = (skill.directory / script_path).resolve()
    root = skill.directory.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise SkillRouterError("script_path must stay inside the skill directory") from exc
    if candidate.suffix != ".py":
        raise SkillRouterError("Only .py skill scripts can be executed")
    if not candidate.exists() or not candidate.is_file():
        raise SkillRouterError(f"Skill script does not exist: {script_path}")
    return candidate.relative_to(root)


def _default_confirmer(
    skill: Skill, command: CommandSpec, arguments: Any
) -> bool:
    print("Command is not in the project whitelist.")
    print(f"Skill: {skill.name}")
    print(f"Command ID: {command.id}")
    print(f"Run template: {command.run}")
    print(f"Arguments: {arguments}")
    answer = input("Add this command to .skill-router/whitelist.json and execute? [y/N] ")
    return answer.strip().lower() in {"y", "yes"}
