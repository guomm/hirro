from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from skill_router.loader import SkillLoader


class LoaderTests(unittest.TestCase):
    def test_loads_skill_command_from_frontmatter(self) -> None:
        with TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            skill_dir = tmp_path / "skills" / "demo"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                """---
name: demo
description: Demo skill
command:
  id: run_demo
  run: python scripts/demo.py
  args:
    text: string
    count: integer
  description: Run demo
---

Body text.
""",
                encoding="utf-8",
            )

            skills = SkillLoader(tmp_path / "skills").load()

            self.assertEqual(len(skills), 1)
            self.assertEqual(skills[0].name, "demo")
            command = skills[0].commands["run_demo"]
            self.assertEqual(command.run, "python scripts/demo.py")
            self.assertEqual(command.args, {"text": "string", "count": "integer"})

    def test_loads_instruction_only_skill_without_command(self) -> None:
        with TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            skill_dir = tmp_path / "skills" / "docs"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                """---
name: docs
description: Documentation workflow.
---

Use MCP tools to search docs.
""",
                encoding="utf-8",
            )

            skills = SkillLoader(tmp_path / "skills").load()

            self.assertEqual(len(skills), 1)
            self.assertEqual(skills[0].name, "docs")
            self.assertEqual(skills[0].commands, {})
            self.assertIn("Use MCP tools", skills[0].body)

if __name__ == "__main__":
    unittest.main()
