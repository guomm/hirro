from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from skill_router.executor import Executor, build_argv, validate_arguments
from skill_router.models import CommandSpec, Skill, SkillRouterError
from skill_router.whitelist import WhitelistEntry, WhitelistStore


def make_skill(path: Path) -> Skill:
    command = CommandSpec(
        id="echo_text",
        run="py -3.12 scripts/echo.py",
        args={"text": "string"},
        description="Echo text",
    )
    return Skill(
        name="echo",
        description="Echo text",
        directory=path,
        body="",
        commands={command.id: command},
    )


class ExecutorTests(unittest.TestCase):
    def test_rejects_unknown_command(self) -> None:
        with TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            executor = Executor(WhitelistStore(tmp_path / "whitelist.json"))

            with self.assertRaisesRegex(SkillRouterError, "Unknown command"):
                executor.execute(make_skill(tmp_path), "missing", {"text": "hi"})

    def test_rejects_invalid_arguments(self) -> None:
        with TemporaryDirectory() as raw:
            command = make_skill(Path(raw)).commands["echo_text"]

            with self.assertRaisesRegex(SkillRouterError, "Unexpected"):
                validate_arguments(command, {"text": "hi", "extra": "no"})

    def test_shell_injection_text_is_passed_as_plain_argument(self) -> None:
        command = CommandSpec(
            id="echo_text",
            run="py -3.12 scripts/echo.py",
            args={"text": "string"},
        )

        argv = build_argv(command, {"text": "hello; rm -rf /"})

        self.assertEqual(argv, ["py", "-3.12", "scripts/echo.py", "--text", "hello; rm -rf /"])

    def test_unwhitelisted_command_can_be_rejected_without_write(self) -> None:
        with TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            whitelist = WhitelistStore(tmp_path / "whitelist.json")
            executor = Executor(whitelist, confirmer=lambda *_: False)

            with self.assertRaisesRegex(SkillRouterError, "cancelled"):
                executor.execute(make_skill(tmp_path), "echo_text", {"text": "hi"})

            self.assertFalse((tmp_path / "whitelist.json").exists())

    def test_whitelisted_command_executes_without_confirmation(self) -> None:
        with TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            skill = make_skill(tmp_path)
            command = skill.commands["echo_text"]
            whitelist = WhitelistStore(tmp_path / "whitelist.json")
            whitelist.add(WhitelistEntry.from_command(skill, command))

            with mock.patch.object(subprocess, "run") as fake_run:
                fake_run.return_value = subprocess.CompletedProcess(
                    args=["py"], returncode=0, stdout="hi\n", stderr=""
                )
                result = Executor(
                    whitelist,
                    confirmer=lambda *_: self.fail("should not confirm"),
                ).execute(skill, "echo_text", {"text": "hi"})

            self.assertEqual(result.returncode, 0)
            self.assertEqual(
                fake_run.call_args.args[0], ["py", "-3.12", "scripts/echo.py", "--text", "hi"]
            )
            self.assertFalse(fake_run.call_args.kwargs["shell"])

    def test_confirmation_adds_whitelist_and_executes(self) -> None:
        with TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            skill = make_skill(tmp_path)
            whitelist = WhitelistStore(tmp_path / "whitelist.json")

            with mock.patch.object(subprocess, "run") as fake_run:
                fake_run.return_value = subprocess.CompletedProcess(
                    args=["py"], returncode=0, stdout="", stderr=""
                )
                Executor(whitelist, confirmer=lambda *_: True).execute(
                    skill, "echo_text", {"text": "hi"}
                )

            self.assertTrue(
                whitelist.contains(WhitelistEntry.from_command(skill, skill.commands["echo_text"]))
            )

    def test_execute_script_runs_relative_python_file(self) -> None:
        with TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            skill = make_skill(tmp_path)
            scripts = tmp_path / "scripts"
            scripts.mkdir()
            (scripts / "helper.py").write_text("print('ok')\n", encoding="utf-8")
            whitelist = WhitelistStore(tmp_path / "whitelist.json")

            with mock.patch.object(subprocess, "run") as fake_run:
                fake_run.return_value = subprocess.CompletedProcess(
                    args=["py"], returncode=0, stdout="ok\n", stderr=""
                )
                result = Executor(whitelist, confirmer=lambda *_: True).execute_script(
                    skill,
                    "scripts/helper.py",
                    ["--mode", "retry-plan"],
                )

            self.assertEqual(result.stdout, "ok\n")
            self.assertEqual(
                fake_run.call_args.args[0],
                ["py", "-3.12", "scripts/helper.py", "--mode", "retry-plan"],
            )
            self.assertFalse(fake_run.call_args.kwargs["shell"])

    def test_execute_script_rejects_path_escape(self) -> None:
        with TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            skill = make_skill(tmp_path)
            executor = Executor(WhitelistStore(tmp_path / "whitelist.json"))

            with self.assertRaisesRegex(SkillRouterError, "inside the skill directory"):
                executor.execute_script(skill, "../outside.py", [])


if __name__ == "__main__":
    unittest.main()
