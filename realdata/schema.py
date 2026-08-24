"""RealDialogue: общая структура, которую производит каждый адаптер в realdata/."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RealDialogue:
    id: str
    texts: list[str] = field(default_factory=list)
    qas: list[dict] = field(default_factory=list)
