from __future__ import annotations

import unittest

from skill_router.models import SkillRouterError
from skill_router.tools import default_tool_registry


class BuiltinToolTests(unittest.TestCase):
    def test_calculator_executes_arithmetic(self) -> None:
        registry = default_tool_registry()

        result = registry.execute("calculator", {"expression": "2 + 3 * 4"})

        self.assertEqual(result["result"], 14)

    def test_calculator_rejects_non_arithmetic(self) -> None:
        registry = default_tool_registry()

        with self.assertRaisesRegex(SkillRouterError, "Invalid calculator"):
            registry.execute("calculator", {"expression": "__import__('os').system('dir')"})

    def test_text_stats(self) -> None:
        registry = default_tool_registry()

        result = registry.execute("text_stats", {"text": "hello world\nagain"})

        self.assertEqual(result, {"characters": 17, "words": 3, "lines": 2})


if __name__ == "__main__":
    unittest.main()
