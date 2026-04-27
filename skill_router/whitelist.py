from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from skill_router.models import CommandSpec, Skill


@dataclass(frozen=True)
class WhitelistEntry:
    skill_name: str
    command_id: str
    run: str

    @classmethod
    def from_command(cls, skill: Skill, command: CommandSpec) -> "WhitelistEntry":
        return cls(skill_name=skill.name, command_id=command.id, run=command.run)

    def to_dict(self) -> dict[str, str]:
        return {
            "skill_name": self.skill_name,
            "command_id": self.command_id,
            "run": self.run,
        }


class WhitelistStore:
    def __init__(self, path: Path | str = ".skill-router/whitelist.json"):
        self.path = Path(path)

    def contains(self, entry: WhitelistEntry) -> bool:
        return entry.to_dict() in self._read_entries()

    def add(self, entry: WhitelistEntry) -> None:
        entries = self._read_entries()
        encoded = entry.to_dict()
        if encoded not in entries:
            entries.append(encoded)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps({"entries": entries}, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    def _read_entries(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text(encoding="utf-8"))
        entries = data.get("entries", [])
        if not isinstance(entries, list):
            return []
        clean: list[dict[str, str]] = []
        for item in entries:
            if (
                isinstance(item, dict)
                and isinstance(item.get("skill_name"), str)
                and isinstance(item.get("command_id"), str)
                and isinstance(item.get("run"), str)
            ):
                clean.append(
                    {
                        "skill_name": item["skill_name"],
                        "command_id": item["command_id"],
                        "run": item["run"],
                    }
                )
        return clean
